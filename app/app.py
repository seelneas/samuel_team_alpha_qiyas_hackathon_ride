"""Addis Ride Demand – hourly forecast demo (deliverable E).

Run from the project root:
    streamlit run app/app.py

The user picks only a zone and a date (1-14 Nov 2025). Weather and events are looked up
from the cleaned tables bundled in app/assets/ (build them with `python app/build_assets.py`).
"""
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
FIRST, LAST = date(2025, 11, 1), date(2025, 11, 14)
TRIPS_PER_DRIVER_DEFAULT = 1.3

st.set_page_config(page_title="Addis Ride Demand Forecast", page_icon="🚕", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    return joblib.load(ASSETS / "final_model.joblib")


@st.cache_data
def load_tables():
    feats = pd.read_csv(ASSETS / "forecast_features.csv", parse_dates=["datetime"])
    weather = pd.read_csv(ASSETS / "weather_forecast.csv", parse_dates=["datetime"])
    events = pd.read_csv(ASSETS / "events_nov.csv", parse_dates=["start", "end"])
    profile = pd.read_csv(ASSETS / "zone_profile.csv")
    stats = pd.read_csv(ASSETS / "zone_stats.csv").set_index("zone")
    return feats, weather, events, profile, stats


def predict(bundle, rows):
    """Apply the saved model to feature rows (handles the optimised multi-seed model)."""
    rows = rows.copy()
    feats = bundle["features"]
    if "profile" in bundle:
        rows = rows.join(bundle["profile"], on=["zone", "day_of_week", "hour"])
    X = rows[feats].copy()
    X["zone"] = pd.Categorical(X["zone"], categories=bundle["zones"])
    models = bundle["model"] if isinstance(bundle["model"], list) else [bundle["model"]]
    return np.clip(np.mean([m.predict(X) for m in models], axis=0), 0, None)


def missing_assets():
    need = ["final_model.joblib", "forecast_features.csv", "weather_forecast.csv",
            "events_nov.csv", "zone_profile.csv", "zone_stats.csv"]
    return [f for f in need if not (ASSETS / f).exists()]


# ---------------------------------------------------------------- UI
st.title("🚕 Addis Ababa ride demand – hourly forecast")
st.caption("Pick a zone and a day. Weather and events are looked up automatically from the bundled cleaned tables.")

if (gone := missing_assets()):
    st.error(f"Missing app assets: {', '.join(gone)}. Run `python app/build_assets.py` from the project root.")
    st.stop()

bundle = load_model()
feats, weather, events, profile, stats = load_tables()
TPD = bundle.get("trips_per_driver", TRIPS_PER_DRIVER_DEFAULT)

c1, c2 = st.columns(2)
zone = c1.selectbox("Zone", sorted(feats.zone.unique()))
day = c2.date_input("Date (1–14 November 2025)", value=FIRST)

if not isinstance(day, date) or not (FIRST <= day <= LAST):
    st.warning(f"Forecasts are only available for **{FIRST:%d %b} – {LAST:%d %b %Y}** "
               "(the period covered by the weather forecast and the trained model). Please pick a date in that range.")
    st.stop()

d0 = pd.Timestamp(day)
d1 = d0 + pd.Timedelta(days=1)
rows = feats[(feats.zone == zone) & (feats.datetime >= d0) & (feats.datetime < d1)].sort_values("datetime")
if len(rows) != 24:
    st.error("No forecast rows for this zone and date in the bundled tables.")
    st.stop()

# ---------------------------------------------------------------- forecast
pred = predict(bundle, rows)
fare = float(stats.loc[zone, "avg_fare_birr"]) if zone in stats.index else np.nan
dow = d0.dayofweek
typical = profile[(profile.zone == zone) & (profile.day_of_week == dow)].set_index("hour").typical_trips \
    .reindex(range(24))
wx = weather[(weather.datetime >= d0) & (weather.datetime < d1)]
wx = wx.set_index(wx.datetime.dt.hour.rename("hour"))

out = pd.DataFrame({
    "hour": [f"{h:02d}:00" for h in range(24)],
    "forecast_trips": pred.round(1),
    "typical_trips": typical.values.round(1),
    "drivers_needed": np.ceil(pred / TPD).astype(int),
    "expected_fares_birr": (pred * fare).round(0),
    "temp_c": wx.temp_c.reindex(range(24)).values,
    "rain_mm": wx.rain_mm.reindex(range(24)).values,
})

# events in this zone (and city-wide ones) that touch this day, incl. the ±2 h window the model uses
win = pd.Timedelta(hours=2)
ev_day = events[(events.start - win < d1) & (events.end + win > d0)] \
    .drop_duplicates(subset=["event_name", "zones", "start", "end"])
zone_ev = ev_day[ev_day.zones.fillna("").str.split("|").apply(lambda z: zone in z or "ALL" in [s.upper() for s in z])]
other_ev = ev_day.drop(zone_ev.index)

peak = int(np.argmax(pred))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total trips (day)", f"{pred.sum():,.0f}",
          f"{100 * (pred.sum() / typical.sum() - 1):+.0f}% vs typical {d0:%A}" if typical.notna().all() else None)
m2.metric("Peak hour", f"{peak:02d}:00", f"{pred[peak]:.0f} trips")
m3.metric("Drivers at peak", f"{int(np.ceil(pred[peak] / TPD))}", f"≈{TPD} trips per driver-hour", delta_color="off")
m4.metric("Expected gross fares", f"{(pred * fare).sum():,.0f} birr", f"avg fare {fare:.0f} birr", delta_color="off")

# ---------------------------------------------------------------- looked up
parts = []
rainy = wx[wx.rain_mm > 0]
if len(rainy):
    top = rainy.rain_mm.idxmax()
    parts.append(f"rain in {len(rainy)} hours, heaviest {rainy.rain_mm.max():.1f} mm at {top:02d}:00 "
                 f"(total {rainy.rain_mm.sum():.1f} mm)")
else:
    parts.append("dry all day")
parts.append(f"temperature {wx.temp_c.min():.0f}–{wx.temp_c.max():.0f} °C")
for _, e in zone_ev.iterrows():
    att = f", ~{e.attendance:,.0f} people" if pd.notna(e.attendance) else ""
    venue = f" at {e.venue}" if pd.notna(e.venue) else ""
    parts.append(f"**{e.event_name}**{venue} {e.start:%d %b %H:%M}–{e.end:%H:%M}{att}")
if zone_ev.empty:
    parts.append("no events in this zone")
st.info("🔎 **Looked up:** " + "; ".join(parts) + ".")
if len(other_ev):
    st.caption("Elsewhere in the city that day: " + "; ".join(
        f"{e.event_name} ({e.zones}, {e.start:%H:%M}–{e.end:%H:%M})" for _, e in other_ev.iterrows()))

# ---------------------------------------------------------------- chart
fig, ax = plt.subplots(figsize=(11, 4))
hrs = np.arange(24)
for _, e in zone_ev.iterrows():
    s = max((e.start - d0) / pd.Timedelta(hours=1), 0)
    t = min((e.end - d0) / pd.Timedelta(hours=1), 24)
    ax.axvspan(s, t, color="#D55E00", alpha=0.15, label=f"event: {e.event_type.replace('_', ' ')}")
for h in wx.index[wx.rain_mm > 0]:
    ax.axvspan(h - 0.5, h + 0.5, color="#56B4E9", alpha=0.12, lw=0)
ax.plot(hrs, typical.values, "--", color="grey", label=f"typical {d0:%A} (last 8 weeks)")
ax.plot(hrs, pred, "-o", color="#0072B2", ms=4, label="forecast")
ax.scatter([peak], [pred[peak]], s=120, color="#E69F00", zorder=5, label=f"peak {peak:02d}:00")
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
ax.set_xlabel("hour (East Africa Time)")
ax.set_ylabel("trips per hour")
ax.set_title(f"{zone} – {d0:%A %d %B %Y}  (blue shading = rain hours)")
h_, l_ = ax.get_legend_handles_labels()
uniq = dict(zip(l_, h_))
ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------------------------------------------------------------- table
st.subheader("Hourly forecast")
st.dataframe(out, hide_index=True, width="stretch")
st.download_button("Download CSV", out.to_csv(index=False), file_name=f"forecast_{zone}_{d0:%Y%m%d}.csv")
st.caption(f"Model: LightGBM (Poisson), trained on data up to {bundle.get('trained_until', '?')}. "
           "Validation RMSE 8.47 trips per zone-hour (MAE 5.85). Drivers = forecast ÷ trips per driver-hour; "
           "fares = forecast × the zone's historical average fare.")
