"""Build the lookup tables the demo app needs, from the cleaned project outputs.

Run once from the project root after notebooks 01 and 04:
    python app/build_assets.py

Writes to app/assets/:
    final_model.joblib       trained model (copied from models/)
    forecast_features.csv    one row per zone-hour for 1-14 Nov 2025 with every model feature
    weather_forecast.csv     cleaned hourly weather forecast for 1-14 Nov (EAT clock)
    events_nov.csv           cleaned events that touch 1-14 Nov
    zone_profile.csv         typical trips per zone x weekday x hour (last 8 weeks of history)
    zone_stats.csv           average fare per zone from the history
"""
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, ASSETS = ROOT / "data" / "processed", Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)

train = pd.read_csv(PROC / "master_train.csv", parse_dates=["datetime"])
test = pd.read_csv(PROC / "master_test.csv", parse_dates=["datetime"])
weather = pd.read_csv(PROC / "weather_hourly_processed.csv", parse_dates=["datetime"])
events = pd.read_csv(PROC / "events_processed.csv", parse_dates=["start", "end"])
WIN_START, WIN_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-15")

# ---- features identical to notebook 04 (D8 + D10) ---------------------------
full = pd.concat([train.assign(_t=1), test.assign(_t=0)], ignore_index=True)
full["rain_class_ord"] = full.rain_class.map({"none": 0, "light": 1, "moderate": 2, "heavy": 3})
key = full.set_index(["zone", "datetime"]).index
for h in (672, 840):
    src = full[["zone", "datetime", "trips"]].assign(datetime=lambda d: d.datetime + pd.Timedelta(hours=h))
    full[f"lag_{h}h"] = key.map(src.set_index(["zone", "datetime"]).trips.to_dict().get)
L = ["lag_336h", "lag_504h", "lag_672h", "lag_840h"]
full[L] = full[L].astype(float)
full["wk_mean"], full["wk_med"], full["wk_std"] = full[L].mean(axis=1), full[L].median(axis=1), full[L].std(axis=1)
full["lag_ratio"] = full.lag_336h / full.lag_504h.clip(lower=1)
full["hour_dow"] = full.hour * 7 + full.day_of_week
hol_days = set()
for _, e in events[(events.event_type == "public_holiday") & (events.status != "cancelled")].iterrows():
    hol_days.update(pd.date_range(e.start.normalize(), e.end.normalize()))
full["is_holiday_eve"] = (full.datetime.dt.normalize() + pd.Timedelta(days=1)).isin(hol_days).astype(int)
full["post_event_attendance"] = full.event_post * full.event_attendance

import joblib  # noqa: E402

bundle = joblib.load(ROOT / "models" / "final_model.joblib")
feats = [f for f in bundle["features"] if f != "prof8"]
fc = full[full._t == 0][["zone", "datetime"] + feats]
missing = set(feats) - set(fc.columns)
assert not missing, f"features missing for the app: {missing}"
fc.to_csv(ASSETS / "forecast_features.csv", index=False)
shutil.copy(ROOT / "models" / "final_model.joblib", ASSETS / "final_model.joblib")

# ---- lookup tables -----------------------------------------------------------
weather[(weather.datetime >= WIN_START) & (weather.datetime < WIN_END)].to_csv(ASSETS / "weather_forecast.csv", index=False)
ev = events[(events.end >= WIN_START - pd.Timedelta(days=1)) & (events.start < WIN_END) & (events.status != "cancelled")]
ev.to_csv(ASSETS / "events_nov.csv", index=False)

recent = train[train.datetime >= train.datetime.max() - pd.Timedelta(weeks=8)]
recent.groupby(["zone", "day_of_week", "hour"]).trips.mean().rename("typical_trips").reset_index() \
      .to_csv(ASSETS / "zone_profile.csv", index=False)
train.groupby("zone").agg(avg_fare_birr=("avg_fare_birr", "mean"), avg_trips=("trips", "mean")).round(2) \
     .reset_index().to_csv(ASSETS / "zone_stats.csv", index=False)

print("assets written:", sorted(p.name for p in ASSETS.iterdir()))
print("forecast rows:", len(fc), "| features:", len(feats))
