"""Make forecasts with a saved model.

    python -m src.predict                                  # submission/team_alpha_submission.csv
    python -m src.predict --out /tmp/sub.csv
    python -m src.predict --zone Bole --date 2025-11-05     # one zone-day, printed

From Python:
    from src.predict import load_model, forecast_zone_day
    forecast_zone_day(load_model(), "Bole", "2025-11-05")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, SUBMISSION_DIR, TEST_END, TEST_START
from .features import add_model_features, load_processed
from .train import predict

DEFAULT_SUBMISSION = SUBMISSION_DIR / "team_alpha_submission.csv"


def load_model(path: Path = MODELS_DIR / "final_model.joblib") -> dict:
    return joblib.load(path)


def forecast_features(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Feature rows for every test zone-hour (1–14 Nov), in template order."""
    tr, te, ev = load_processed(processed_dir)
    _, test = add_model_features(tr, te, ev)
    return test


def make_submission(bundle: dict, out_path: Path = DEFAULT_SUBMISSION, processed_dir: Path = PROCESSED_DIR,
                    raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    test = forecast_features(processed_dir)
    test["predicted_trips"] = predict(bundle, test)
    tmpl = pd.read_csv(raw_dir / "submission_template.csv")
    sub = tmpl[["row_id"]].merge(test[["row_id", "predicted_trips"]], on="row_id", how="left")
    assert len(sub) == 4032 and sub.row_id.is_unique and sub.row_id.tolist() == tmpl.row_id.tolist()
    assert sub.predicted_trips.notna().all() and (sub.predicted_trips >= 0).all()
    sub["predicted_trips"] = sub.predicted_trips.round(3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"submission {sub.shape} -> {out_path} | mean predicted {sub.predicted_trips.mean():.2f}")
    return sub


def forecast_zone_day(bundle: dict, zone: str, day, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """24-hour forecast for one zone and date in 1–14 Nov 2025, with drivers needed."""
    d0 = pd.Timestamp(day).normalize()
    if not (TEST_START <= d0 <= TEST_END):
        raise ValueError(f"date must be between {TEST_START:%d %b} and {TEST_END:%d %b %Y}")
    feats = features if features is not None else forecast_features()
    rows = feats[(feats.zone == zone) & (feats.datetime.dt.normalize() == d0)].sort_values("datetime")
    if rows.empty:
        raise ValueError(f"unknown zone {zone!r}")
    p = predict(bundle, rows)
    tpd = bundle.get("trips_per_driver", 1.3)
    return pd.DataFrame({"datetime": rows.datetime.values, "forecast_trips": p.round(1),
                         "drivers_needed": np.ceil(p / tpd).astype(int),
                         "rain_mm": rows.rain_mm.values, "event_in_window": rows.event_in_window.values})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "final_model.joblib")
    ap.add_argument("--out", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--zone")
    ap.add_argument("--date")
    a = ap.parse_args()
    m = load_model(a.model)
    if a.zone and a.date:
        print(forecast_zone_day(m, a.zone, a.date).to_string(index=False))
    else:
        make_submission(m, a.out)
