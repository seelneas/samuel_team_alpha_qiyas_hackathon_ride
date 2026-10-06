"""Train, validate and save the final model (notebook 04, D3/D6/D10 + final fit).

    python -m src.train                    # validate + retrain on all history -> models/final_model.joblib
    python -m src.train --skip-cv          # only retrain the final model
    python -m src.train --out /tmp/m.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CANONICAL_ZONES, HORIZON_DAYS, MODELS_DIR, PROCESSED_DIR, ROLLING_CUTS, SEED, VAL_START
from .features import (FEATURES, PROFILE, add_model_features, attach_profile, check_no_leakage, load_processed,
                       mae, recent_profile, rmse, to_X)

BASE_LGB = dict(subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=-1, random_state=SEED)
FINAL_PARAMS = dict(objective="poisson", num_leaves=15, learning_rate=0.05, n_estimators=1500,
                    min_child_samples=20, colsample_bytree=0.6, reg_lambda=5.0)
SEEDS = (1, 2, 3)


def fit(train_slice: pd.DataFrame, features: list[str] = FEATURES, params: dict = FINAL_PARAMS,
        seeds=SEEDS) -> dict:
    """Fit the seed-averaged Poisson LightGBM. Returns the model bundle used by predict()."""
    check_no_leakage(features)
    prof = recent_profile(train_slice)
    t = attach_profile(train_slice, prof)
    feats = [f for f in features if f != PROFILE] + ([PROFILE] if PROFILE in features else [])
    models = [lgb.LGBMRegressor(**{**BASE_LGB, **params, "random_state": s}).fit(to_X(t, feats), t.trips)
              for s in seeds]
    return {"kind": "lgb_v2", "model": models, "features": feats, "profile": prof, "params": params,
            "zones": sorted(CANONICAL_ZONES), "trained_until": str(train_slice.datetime.max())}


def predict(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    d = attach_profile(df, bundle["profile"]) if "profile" in bundle else df
    models = bundle["model"] if isinstance(bundle["model"], list) else [bundle["model"]]
    X = to_X(d, bundle["features"], bundle.get("zones", CANONICAL_ZONES))
    return np.clip(np.mean([m.predict(X) for m in models], axis=0), 0, None)


def split(df: pd.DataFrame, cut: pd.Timestamp, days: int = HORIZON_DAYS):
    return df[df.datetime < cut], df[(df.datetime >= cut) & (df.datetime < cut + pd.Timedelta(days=days))]


def seasonal_naive(tr: pd.DataFrame, va: pd.DataFrame) -> np.ndarray:
    m = tr.groupby(["zone", "day_of_week", "hour"]).trips.mean().rename("p")
    return va.join(m, on=["zone", "day_of_week", "hour"])["p"].fillna(tr.trips.mean()).values


def evaluate(train: pd.DataFrame, **fit_kw) -> pd.DataFrame:
    """Main split + rolling-origin folds, model vs seasonal-naive."""
    rows = []
    for cut in ROLLING_CUTS + ([VAL_START] if VAL_START not in ROLLING_CUTS else []):
        tr, va = split(train, cut)
        p = predict(fit(tr, **fit_kw), va)
        rows.append({"cut": cut.date(), "rmse": rmse(va.trips, p), "mae": mae(va.trips, p),
                     "seasonal_naive_rmse": rmse(va.trips, seasonal_naive(tr, va)), "n": len(va)})
        print(f"  fold {cut:%d %b}: RMSE {rows[-1]['rmse']:.3f}  (seasonal-naive {rows[-1]['seasonal_naive_rmse']:.3f})")
    return pd.DataFrame(rows)


def train_final(processed_dir: Path = PROCESSED_DIR, out_path: Path = MODELS_DIR / "final_model.joblib",
                run_cv: bool = True) -> dict:
    tr, te, ev = load_processed(processed_dir)
    train, _ = add_model_features(tr, te, ev)
    if run_cv:
        print("Rolling-origin validation:")
        res = evaluate(train)
        print(f"  mean RMSE {res.rmse.mean():.3f} ± {res.rmse.std(ddof=0):.3f} | "
              f"main split ({VAL_START:%d %b}) RMSE {res.iloc[-1].rmse:.3f}, MAE {res.iloc[-1].mae:.3f}")
    bundle = fit(train)
    bundle["trips_per_driver"] = round(float((tr.trips / tr.active_drivers).median()), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"model saved -> {out_path}")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "final_model.joblib")
    ap.add_argument("--skip-cv", action="store_true")
    a = ap.parse_args()
    train_final(out_path=a.out, run_cv=not a.skip_cv)
