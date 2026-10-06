"""Model features on top of the master tables (notebook 04, sections D4, D8 and D10).

All features are known at forecast time: calendar, weather forecast, scheduled events,
and trip history that is at least 14 days old (the forecast horizon).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CANONICAL_ZONES

CALENDAR = ["zone", "hour", "day_of_week", "is_weekend", "day_of_month", "is_payday_window",
            "is_public_holiday", "is_school_break", "days_since_start", "hour_sin", "hour_cos"]
LAGS = ["lag_336h", "lag_504h", "roll_mean_168h_lag336"]
WEATHER = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh", "rain_3h_sum", "is_raining", "rain_class_ord"]
EVENTS = ["event_in_window", "event_attendance", "football_window", "concert_window", "conference_window",
          "road_closure_window", "event_pre", "event_during", "event_post"]
RESPONSE = ["is_holiday_eve", "post_event_attendance"]                          # D8
LONG_LAGS = ["lag_672h", "lag_840h", "wk_mean", "wk_med", "wk_std", "lag_ratio", "hour_dow"]  # D10
PROFILE = "prof8"                                                               # D10, fitted per training slice

FEATURES = CALENDAR + LAGS + WEATHER + EVENTS + RESPONSE + LONG_LAGS + [PROFILE]

RAIN_ORD = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}
EXCLUDED = {"active_drivers", "avg_wait_min", "avg_fare_birr", "month", "record_id", "row_id", "pickup_hour"}


def holiday_days(events: pd.DataFrame) -> set:
    days = set()
    for _, e in events[(events.event_type == "public_holiday") & (events.status != "cancelled")].iterrows():
        days.update(pd.date_range(e.start.normalize(), e.end.normalize()))
    return days


def add_model_features(train: pd.DataFrame, test: pd.DataFrame, events: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add every model feature except the profile (which depends on the training slice).

    train/test are the master tables; events is events_processed.csv (parsed start/end)."""
    full = pd.concat([train.assign(_t=1), test.assign(_t=0)], ignore_index=True)
    full["rain_class_ord"] = full["rain_class"].map(RAIN_ORD)

    hol = holiday_days(events)
    full["is_holiday_eve"] = (full.datetime.dt.normalize() + pd.Timedelta(days=1)).isin(hol).astype(int)
    full["post_event_attendance"] = full.event_post * full.event_attendance

    key = full.set_index(["zone", "datetime"]).index
    for h in (672, 840):
        src = full[["zone", "datetime", "trips"]].assign(datetime=lambda d: d.datetime + pd.Timedelta(hours=h))
        full[f"lag_{h}h"] = key.map(src.set_index(["zone", "datetime"]).trips.to_dict().get)
    L = ["lag_336h", "lag_504h", "lag_672h", "lag_840h"]
    full[L] = full[L].astype(float)
    full["wk_mean"] = full[L].mean(axis=1)
    full["wk_med"] = full[L].median(axis=1)
    full["wk_std"] = full[L].std(axis=1)
    full["lag_ratio"] = full.lag_336h / full.lag_504h.clip(lower=1)
    full["hour_dow"] = full.hour * 7 + full.day_of_week

    tr = full[full._t == 1].drop(columns="_t").reset_index(drop=True)
    te = full[full._t == 0].drop(columns=["_t", "trips"], errors="ignore").reset_index(drop=True)
    return tr, te


def recent_profile(train_slice: pd.DataFrame, weeks: int = 8) -> pd.Series:
    """Mean trips per zone x weekday x hour over the last `weeks` weeks of a training slice."""
    rec = train_slice[train_slice.datetime >= train_slice.datetime.max() - pd.Timedelta(weeks=weeks)]
    return rec.groupby(["zone", "day_of_week", "hour"]).trips.mean().rename(PROFILE)


def attach_profile(df: pd.DataFrame, profile: pd.Series) -> pd.DataFrame:
    return df.drop(columns=PROFILE, errors="ignore").join(profile, on=["zone", "day_of_week", "hour"])


def to_X(df: pd.DataFrame, features: list[str], zones: list[str] = CANONICAL_ZONES) -> pd.DataFrame:
    X = df[features].copy()
    if "zone" in X:
        X["zone"] = pd.Categorical(X["zone"], categories=sorted(zones))
    return X


def check_no_leakage(features: list[str]) -> None:
    bad = set(features) & EXCLUDED
    assert not bad, f"features not available at forecast time: {bad}"
    assert any(f in features for f in WEATHER) and any(f in features for f in EVENTS), \
        "Rule 5: model must use weather and event features"


def load_processed(processed_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(processed_dir / "master_train.csv", parse_dates=["datetime"])
    te = pd.read_csv(processed_dir / "master_test.csv", parse_dates=["datetime"])
    ev = pd.read_csv(processed_dir / "events_processed.csv", parse_dates=["start", "end"])
    return tr, te, ev


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
