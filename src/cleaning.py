"""Deliverable A as a reusable pipeline: clean the raw tables, join them and export the master tables.

Mirrors notebooks/01_cleaning_and_integration.ipynb step for step (same rules, same outputs).

    python -m src.cleaning                     # writes to data/processed/ and reports/
    python -m src.cleaning --out /tmp/check    # write somewhere else (e.g. to compare)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (CANONICAL_ZONES, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, TEST_END, TEST_START,
                     TRAIN_END, TRAIN_START, TZ_LOCAL)

# --------------------------------------------------------------------------- cleaning log
class CleaningLog(list):
    def add(self, file, columns, issue, n_rows, total, fix, why):
        self.append({"file": file, "columns": columns, "issue_type": issue, "rows_affected": int(n_rows),
                     "pct_rows": round(100 * n_rows / total, 2), "fix_applied": fix, "why": why})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self)


# --------------------------------------------------------------------------- timestamps (A2b/A2c)
FORMATS = [
    ("iso_offset", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",
     lambda s: pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%S%z", utc=True)),
    ("iso_utc_Z", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
     lambda s: pd.to_datetime(s, format="%Y-%m-%dT%H:%M:%SZ").dt.tz_localize("UTC")),
    ("ymd_hm", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", lambda s: pd.to_datetime(s, format="%Y-%m-%d %H:%M")),
    ("dmy_hm", r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$", lambda s: pd.to_datetime(s, format="%d/%m/%Y %H:%M")),
    ("mon_d_y_ampm", r"^[A-Z][a-z]{2} \d{2}, \d{4} \d{2}:\d{2} [AP]M$",
     lambda s: pd.to_datetime(s, format="%b %d, %Y %I:%M %p")),
]


def parse_mixed(series: pd.Series, naive_tz: str) -> tuple[pd.Series, pd.Series]:
    """Parse a mixed-format text column into tz-naive EAT timestamps.

    naive_tz: clock used by strings without an offset ('EAT' or 'UTC').
    Returns (timestamps, detected_format_label)."""
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    fmt = pd.Series("unparsed", index=s.index, dtype=object)
    for label, rx, parser in FORMATS:
        m = s.str.match(rx).fillna(False)
        if not m.any():
            continue
        p = parser(s[m])
        if getattr(p.dt, "tz", None) is not None:
            p = p.dt.tz_convert(TZ_LOCAL).dt.tz_localize(None)
        elif naive_tz == "UTC":
            p = p.dt.tz_localize("UTC").dt.tz_convert(TZ_LOCAL).dt.tz_localize(None)
        out[m] = p.astype("datetime64[ns]")
        fmt[m] = label
    return out, fmt


# --------------------------------------------------------------------------- zones / event types (A2a)
ZONE_ALIASES = {"bole rd": "Bole", "mercato": "Merkato", "kazanches": "Kazanchis", "megenaga": "Megenagna",
                "piazza": "Piassa", "kolfe keranio": "Kolfe", "c.m.c": "CMC", "cmc": "CMC"}
CITYWIDE = {"citywide", "city-wide", "all", "all zones"}
_CANON_LOWER = {z.lower(): z for z in CANONICAL_ZONES}
EVENT_TYPES = ["public_holiday", "school_break", "football_match", "concert",
               "conference", "exhibition", "road_closure", "sports_run"]
WINDOW_TYPES = ["football_match", "concert", "conference", "exhibition", "road_closure", "sports_run"]


def clean_zone(value):
    if pd.isna(value):
        return np.nan
    v = re.sub(r"\s+", " ", str(value)).strip().lower()
    v = re.sub(r"\s*\(.*?\)", "", v).strip()
    v = ZONE_ALIASES.get(v, v)
    return _CANON_LOWER.get(str(v).lower(), np.nan)


def clean_event_zones(value) -> list[str]:
    if pd.isna(value):
        return []
    v = re.sub(r"\s+", " ", str(value)).strip().lower()
    if v in CITYWIDE:
        return list(CANONICAL_ZONES)
    return [z for z in (clean_zone(p) for p in re.split(r"&|,|/| and ", v)) if isinstance(z, str)]


def clean_event_type(v):
    t = re.sub(r"[\s\-]+", "_", str(v).strip().lower())
    return t if t in EVENT_TYPES else np.nan


# --------------------------------------------------------------------------- loading
def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(raw_dir / "ride_demand_train.csv", dtype={"pickup_hour": str, "zone": str}),
        "test": pd.read_csv(raw_dir / "ride_demand_test.csv", dtype={"pickup_hour": str, "zone": str}),
        "weather": pd.read_csv(raw_dir / "weather_hourly.csv", dtype={"timestamp": str}),
        "events": pd.read_csv(raw_dir / "events_calendar.csv", dtype=str),
        "template": pd.read_csv(raw_dir / "submission_template.csv"),
    }


# --------------------------------------------------------------------------- trips
SPIKE_RATIO = 4.0  # > 4 trips per active driver in one hour is not physically plausible


def clean_trips(train_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    N = len(train_raw)
    trips = train_raw.copy()
    trips["datetime"], trips["ts_format"] = parse_mixed(trips["pickup_hour"], naive_tz="EAT")
    assert trips["datetime"].notna().all()
    log.add("ride_demand_train", "pickup_hour", "mixed timestamp formats (3)", (trips.ts_format != "ymd_hm").sum(), N,
            "regex-detected explicit formats; +03:00 converted to EAT", "avoid day/month swaps and clock drift")

    trips["zone_raw"] = trips["zone"]
    trips["zone"] = trips["zone_raw"].map(clean_zone)
    assert trips["zone"].notna().all()
    log.add("ride_demand_train", "zone", "inconsistent zone spelling (55 variants)",
            (trips.zone_raw != trips.zone).sum(), N, "case/space normalization + alias map to 12 labels",
            "one key per zone across all tables")

    dup = trips.duplicated(subset=["zone", "datetime"], keep="first")
    log.add("ride_demand_train", "zone, pickup_hour", "duplicate zone-hour rows", dup.sum(), N,
            "kept first occurrence", "exactly one row per zone-hour")
    trips = trips[~dup].copy()

    sent = trips["trips"].eq(-1)
    log.add("ride_demand_train", "trips", "sentinel -1 (no reading)", sent.sum(), N,
            "set to NaN, row excluded from training target", "-1 trips is impossible")
    trips.loc[sent, "trips"] = np.nan
    log.add("ride_demand_train", "trips", "missing target", trips["trips"].isna().sum() - sent.sum(), N,
            "row excluded from training target (kept in grid for lags)", "cannot learn from unknown target")

    spike = (trips["trips"] / trips["active_drivers"]) > SPIKE_RATIO
    log.add("ride_demand_train", "trips", "data-entry spikes (trips > 4x active drivers)", spike.sum(), N,
            "set to NaN", "e.g. 640 trips with 56 drivers; real event peaks keep the driver ratio")
    trips.loc[spike, "trips"] = np.nan

    wait_bad = trips["avg_wait_min"].lt(0)
    log.add("ride_demand_train", "avg_wait_min", "sentinel -1", wait_bad.sum(), N, "set to NaN",
            "negative wait impossible (analysis-only column)")
    trips.loc[wait_bad, "avg_wait_min"] = np.nan
    log.add("ride_demand_train", "avg_fare_birr", "missing values", trips["avg_fare_birr"].isna().sum(), N,
            "left NaN (not a model input)", "operational column, analysis only")
    return trips


def gap_grid(trips: pd.DataFrame, log: CleaningLog) -> tuple[pd.DataFrame, pd.Series]:
    """Zone-hour grid labelled pre-launch / outage / random-missing. Returns (grid, launch hour per zone)."""
    launch = trips.groupby("zone")["datetime"].min().rename("first_hour")
    full_hours = pd.date_range(TRAIN_START, TRAIN_END, freq="h")
    grid = pd.MultiIndex.from_product([CANONICAL_ZONES, full_hours], names=["zone", "datetime"]).to_frame(index=False)
    grid = grid.merge(launch.reset_index(), on="zone")
    grid["pre_launch"] = grid["datetime"] < grid["first_hour"]
    hour_rows = trips.groupby("datetime").size().reindex(full_hours, fill_value=0)
    outage_hours = hour_rows[hour_rows == 0].index
    grid["outage"] = grid["datetime"].isin(outage_hours)
    log.add("ride_demand_train", "(rows)", "platform outage – whole city missing", len(outage_hours), len(full_hours),
            "not imputed; hours excluded from training", "no demand was recorded, imputing would invent data")
    log.add("ride_demand_train", "(rows)", "late-launching zone (Ayat) – no rows before launch",
            grid.pre_launch.sum(), len(grid), "zone-hours before launch excluded", "zone did not exist; not zero demand")
    return grid, launch


# --------------------------------------------------------------------------- weather
def clean_weather(weather_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    NW = len(weather_raw)
    w = weather_raw.copy()
    w["datetime"], w["ts_format"] = parse_mixed(w["timestamp"], naive_tz="UTC")
    assert w["datetime"].notna().all()
    log.add("weather_hourly", "timestamp", "UTC clock + 2 formats", NW, NW,
            "parsed explicitly, converted UTC -> EAT (+3h)", "trip table is on EAT (see A2c proof)")
    w["data_type"] = w["data_type"].str.strip().str.lower()

    sent = w["rain_mm"].eq(-9999)
    log.add("weather_hourly", "rain_mm", "sentinel -9999", sent.sum(), NW, "set to NaN, later filled",
            "sentinel code for no reading")
    w.loc[sent, "rain_mm"] = np.nan

    fahr = w["temp_c"] > 40
    log.add("weather_hourly", "temp_c", "values in Fahrenheit for part of July", fahr.sum(), NW,
            "converted (F-32)*5/9", "40-77 'C is impossible in Addis; converted values match neighbours")
    w.loc[fahr, "temp_c"] = (w.loc[fahr, "temp_c"] - 32) * 5 / 9

    log.add("weather_hourly", "timestamp", "duplicate hours with conflicting values",
            w.duplicated(subset=["datetime"]).sum(), NW, "averaged numeric values per hour",
            "many-to-one join needs one row per hour")
    num = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]
    w = w.groupby("datetime", as_index=False).agg({**{c: "mean" for c in num}, "data_type": "first"})

    full_w = pd.date_range(TRAIN_START, TEST_END, freq="h")
    log.add("weather_hourly", "timestamp", "missing hours (gaps)", len(full_w.difference(w["datetime"])), len(full_w),
            "reindexed to full hourly grid; interpolated (≤6h), rain filled 0", "every zone-hour needs weather")
    log.add("weather_hourly", "temp_c, humidity_pct", "missing values",
            w[["temp_c", "humidity_pct"]].isna().any(axis=1).sum(), NW, "time interpolation", "smooth variables")
    w = w.set_index("datetime").reindex(full_w)
    w.index.name = "datetime"
    w["weather_imputed"] = w["temp_c"].isna() | w["rain_mm"].isna()
    for c in ["temp_c", "humidity_pct", "wind_kmh"]:
        w[c] = w[c].interpolate(method="time", limit=6, limit_direction="both")
    hist = w.index < TEST_START
    for c in ["temp_c", "humidity_pct", "wind_kmh"]:   # long gaps -> hour-of-day climatology (history only)
        clim = w.loc[hist, c].groupby(w.index[hist].hour).mean()
        w[c] = w[c].fillna(pd.Series(w.index.hour.map(clim), index=w.index))
    w["rain_mm"] = w["rain_mm"].fillna(0.0)
    w["data_type"] = w["data_type"].fillna(
        pd.Series(np.where(w.index >= TEST_START, "forecast", "observed"), index=w.index))
    w = w.reset_index()

    w["rain_3h_sum"] = w["rain_mm"].rolling(3, min_periods=1).sum()
    w["is_raining"] = (w["rain_mm"] > 0).astype(int)
    w["rain_class"] = pd.cut(w["rain_mm"], [-0.01, 0, 2.5, 7.6, np.inf],
                             labels=["none", "light", "moderate", "heavy"]).astype(str)
    return w


# --------------------------------------------------------------------------- events
def clean_events(events_raw: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    NE = len(events_raw)
    ev = events_raw.copy()
    ev["event_type_clean"] = ev["event_type"].map(clean_event_type)
    log.add("events_calendar", "event_type", "inconsistent spelling (20 variants)",
            (ev.event_type != ev.event_type_clean).sum(), NE, "normalized to 8 labels", "group events by type")
    ev["status"] = ev["status"].str.strip().str.lower()
    log.add("events_calendar", "status", "case/whitespace variants", (events_raw.status != ev.status).sum(), NE,
            "lower + strip", "reliable cancelled filter")
    ev["zones"] = ev["zone"].map(clean_event_zones)
    log.add("events_calendar", "zone", "spelling, extra text, multi-zone and city-wide values",
            (ev["zone"].map(clean_zone) != ev["zone"]).sum(), NE,
            "split on '&', strip '(...)', alias map, citywide -> all 12", "interval join needs canonical zone keys")

    ev["start"], ev["start_fmt"] = parse_mixed(ev["start_datetime"], naive_tz="EAT")
    ev["end"], ev["end_fmt"] = parse_mixed(ev["end_datetime"], naive_tz="EAT")
    assert ev["start"].notna().all()
    log.add("events_calendar", "start_datetime, end_datetime", "3 timestamp formats incl. 'Feb 06, 2025 08:00 PM'",
            (ev.start_fmt != "ymd_hm").sum() + (ev.end_fmt != "ymd_hm").sum(), 2 * NE,
            "explicit regex formats", "no day/month confusion")

    med_dur = (ev["end"] - ev["start"])[ev["end"] > ev["start"]].groupby(ev["event_type_clean"]).median()
    bad_end = ev["end"].isna() | (ev["end"] <= ev["start"])
    log.add("events_calendar", "end_datetime", "missing or earlier than start", bad_end.sum(), NE,
            "replaced by start + median duration of that event type", "interval needs a valid end")
    ev.loc[bad_end, "end"] = ev.loc[bad_end, "start"] + ev.loc[bad_end, "event_type_clean"].map(med_dur)

    att = (ev["expected_attendance"].str.lower().str.replace("approx", "", regex=False)
           .str.replace(",", "", regex=False).str.strip())
    ev["attendance"] = pd.to_numeric(att, errors="coerce")
    log.add("events_calendar", "expected_attendance", "free text ('approx 34000', '34,756') and blanks",
            ev["expected_attendance"].notna().sum(), NE, "parsed to number; blanks -> 0 in features",
            "usable numeric crowd size")
    log.add("events_calendar", "status", "cancelled events", ev.status.eq("cancelled").sum(), NE,
            "excluded from event features (kept in file for B3.4)", "did not take place")
    return ev[["event_id", "event_name", "event_type_clean", "venue", "zone", "zones", "start", "end",
               "attendance", "status"]].rename(columns={"event_type_clean": "event_type"})


# --------------------------------------------------------------------------- event interval join (A3)
PRE_H, POST_H = 2, 2
WEATHER_FEATS = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh", "rain_3h_sum", "is_raining", "rain_class"]
EVENT_FEATS = ["event_in_window", "event_attendance", "football_window", "concert_window",
               "conference_window", "road_closure_window", "event_pre", "event_during", "event_post"]
LAG_FEATS = ["lag_336h", "lag_504h", "roll_mean_168h_lag336"]


def explode_events(events_clean: pd.DataFrame) -> pd.DataFrame:
    """One row per (zone, hour, event) from start-2h to end+2h, with phase pre/during/post."""
    active = events_clean[events_clean["status"] != "cancelled"]
    rows = []
    for _, e in active[active.event_type.isin(WINDOW_TYPES)].iterrows():
        s, en = e["start"].floor("h"), e["end"].ceil("h")
        for h in pd.date_range(s - pd.Timedelta(hours=PRE_H), en + pd.Timedelta(hours=POST_H), freq="h"):
            phase = "pre" if h < s else ("post" if h >= en else "during")
            for z in e["zones"]:
                rows.append((z, h, e["event_id"], e["event_type"], phase, e["attendance"],
                             (s - h) / pd.Timedelta(hours=1), (h - en) / pd.Timedelta(hours=1)))
    return pd.DataFrame(rows, columns=["zone", "datetime", "event_id", "event_type", "phase",
                                       "attendance", "hours_to_start", "hours_since_end"])


def event_features(ev_hours: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exploded events to one row per zone-hour."""
    f = ev_hours.groupby(["zone", "datetime"]).agg(
        event_in_window=("event_id", "size"),
        event_attendance=("attendance", "max"),
        football_window=("event_type", lambda s: int((s == "football_match").any())),
        concert_window=("event_type", lambda s: int((s == "concert").any())),
        conference_window=("event_type", lambda s: int((s == "conference").any())),
        road_closure_window=("event_type", lambda s: int((s == "road_closure").any())),
        event_pre=("phase", lambda s: int((s == "pre").any())),
        event_during=("phase", lambda s: int((s == "during").any())),
        event_post=("phase", lambda s: int((s == "post").any())),
    ).reset_index()
    f["event_in_window"] = (f["event_in_window"] > 0).astype(int)
    f["event_attendance"] = f["event_attendance"].fillna(0)
    return f


def event_days(events_clean: pd.DataFrame, types: list[str]) -> set:
    active = events_clean[events_clean["status"] != "cancelled"]
    days = set()
    for _, e in active[active.event_type.isin(types)].iterrows():
        days.update(pd.date_range(e["start"].normalize(), e["end"].normalize(), freq="D"))
    return days


# --------------------------------------------------------------------------- features + master tables (A6/A8)
def add_calendar(df: pd.DataFrame, holiday_days: set, school_days: set) -> pd.DataFrame:
    t = df["datetime"]
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = t.dt.month
    df["day_of_month"] = t.dt.day
    df["is_payday_window"] = ((t.dt.day >= 25) | (t.dt.day <= 3)).astype(int)
    df["is_public_holiday"] = t.dt.normalize().isin(holiday_days).astype(int)
    df["is_school_break"] = t.dt.normalize().isin(school_days).astype(int)
    df["days_since_start"] = (t - TRAIN_START) / pd.Timedelta(days=1)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def join_weather_events(df: pd.DataFrame, weather_clean: pd.DataFrame, ev_feat: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.merge(weather_clean[["datetime"] + WEATHER_FEATS], on="datetime", how="left", validate="m:1")
    df = df.merge(ev_feat, on=["zone", "datetime"], how="left", validate="m:1")
    df[EVENT_FEATS] = df[EVENT_FEATS].fillna(0)
    assert len(df) == n0, "row count changed by join"
    return df


def lag_table(trips: pd.DataFrame, grid: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """Trip lags >= 14 days (the forecast horizon), computed on the full zone-hour grid incl. test hours."""
    base = grid.loc[~grid.pre_launch, ["zone", "datetime"]].merge(
        trips[["zone", "datetime", "trips"]], how="left", on=["zone", "datetime"])
    test_keys = test_raw.assign(datetime=parse_mixed(test_raw["pickup_hour"], "EAT")[0],
                                zone=test_raw["zone"].map(clean_zone))[["zone", "datetime"]]
    lf = pd.concat([base, test_keys.assign(trips=np.nan)], ignore_index=True)
    lf = lf.sort_values(["zone", "datetime"]).reset_index(drop=True)
    g = lf.groupby("zone")["trips"]
    lf["lag_336h"] = g.shift(336)
    lf["lag_504h"] = g.shift(504)
    lf["roll_mean_168h_lag336"] = g.transform(lambda s: s.shift(336).rolling(168, min_periods=24).mean())
    return lf


def build_master_tables(raw: dict, log: CleaningLog | None = None) -> dict:
    """Full A-pipeline. Returns dict with master_train, master_test, weather, events, ev_hours, log."""
    log = log if log is not None else CleaningLog()
    trips = clean_trips(raw["train"], log)
    grid, launch = gap_grid(trips, log)
    weather = clean_weather(raw["weather"], log)
    events = clean_events(raw["events"], log)
    ev_hours = explode_events(events)
    ev_feat = event_features(ev_hours)
    hol, school = event_days(events, ["public_holiday"]), event_days(events, ["school_break"])
    lags = lag_table(trips, grid, raw["test"])

    mt = trips.loc[trips["trips"].notna(),
                   ["record_id", "zone", "datetime", "trips", "avg_fare_birr", "avg_wait_min", "active_drivers"]].copy()
    mt = mt.merge(launch.reset_index(), on="zone")
    mt = mt[mt["datetime"] >= mt.pop("first_hour")]
    n_before = len(mt)
    mt = join_weather_events(add_calendar(mt, hol, school), weather, ev_feat)
    mt = mt.merge(lags[["zone", "datetime"] + LAG_FEATS], on=["zone", "datetime"], how="left", validate="1:1")
    master_train = mt.sort_values(["datetime", "zone"]).reset_index(drop=True)

    te = raw["test"].copy()
    te["datetime"], _ = parse_mixed(te["pickup_hour"], naive_tz="EAT")
    te["zone"] = te["zone"].map(clean_zone)
    te = join_weather_events(add_calendar(te, hol, school), weather, ev_feat)
    te = te.merge(lags[["zone", "datetime"] + LAG_FEATS], on=["zone", "datetime"], how="left", validate="1:1")
    drop = ["record_id", "zone", "datetime", "trips", "avg_fare_birr", "avg_wait_min", "active_drivers"]
    master_test = te[["row_id", "pickup_hour", "zone", "datetime"] +
                     [c for c in master_train.columns if c not in drop]]

    return {"master_train": master_train, "master_test": master_test, "weather": weather, "events": events,
            "ev_hours": ev_hours, "log": log, "n_train_before_join": n_before, "template": raw["template"],
            "n_test_raw": len(raw["test"])}


# --------------------------------------------------------------------------- integrity checks (A7)
NON_FEATURES = {"record_id", "row_id", "pickup_hour", "datetime", "trips", "avg_fare_birr", "avg_wait_min",
                "active_drivers"}
LEAKY = {"avg_fare_birr", "avg_wait_min", "active_drivers", "lag_1h", "lag_24h"}


def validate(out: dict, verbose: bool = True) -> bool:
    tr, te, tmpl = out["master_train"], out["master_test"], out["template"]
    ft = [c for c in tr.columns if c not in NON_FEATURES]
    fs = [c for c in te.columns if c not in NON_FEATURES]
    checks = {
        "one row per zone-hour (train)": not tr.duplicated(["zone", "datetime"]).any(),
        "one row per zone-hour (test)": not te.duplicated(["zone", "datetime"]).any(),
        "only the 12 canonical zones": set(tr.zone) | set(te.zone) == set(CANONICAL_ZONES),
        "train timestamps within 1 Jan–31 Oct (EAT)": tr.datetime.between(TRAIN_START, TRAIN_END).all(),
        "test timestamps within 1–14 Nov (EAT)": te.datetime.between(TEST_START, TEST_END).all(),
        "test has 4,032 rows in template order": len(te) == 4032 and te.row_id.tolist() == tmpl.row_id.tolist(),
        "no negative / sentinel trips": (tr.trips >= 0).all(),
        "no sentinel weather values": (tr.rain_mm >= 0).all() and tr.temp_c.between(-5, 40).all()
                                      and (te.rain_mm >= 0).all(),
        "no missing weather after join": tr[WEATHER_FEATS].notna().all().all() and te[WEATHER_FEATS].notna().all().all(),
        "joins did not change row counts": len(tr) == out["n_train_before_join"] and len(te) == out["n_test_raw"],
        "train and test share the same feature columns": ft == fs,
        "no forecast-time-unavailable feature in test": not (set(fs) & LEAKY),
        "weather & event features present (Rule 5)": "rain_mm" in fs and "event_in_window" in fs,
    }
    if verbose:
        for name, ok in checks.items():
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


# --------------------------------------------------------------------------- export
def data_dictionary(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    source = {**{c: "trips" for c in ["record_id", "zone", "datetime", "trips", "avg_fare_birr", "avg_wait_min",
                                      "active_drivers", "row_id", "pickup_hour"]},
              **{c: "weather" for c in WEATHER_FEATS},
              **{c: "events" for c in EVENT_FEATS + ["is_public_holiday", "is_school_break"]},
              **{c: "trips (history)" for c in LAG_FEATS}}
    rows = []
    for c in tr.columns.union(te.columns, sort=False):
        rows.append({"column": c, "dtype": str((tr if c in tr else te)[c].dtype),
                     "source": source.get(c, "calendar (derived from datetime)"),
                     "in_train": c in tr, "in_test": c in te,
                     "known_at_forecast_time": c not in {"trips", "avg_fare_birr", "avg_wait_min", "active_drivers"}})
    return pd.DataFrame(rows)


def export(out: dict, processed_dir: Path = PROCESSED_DIR, reports_dir: Path = REPORTS_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out["master_train"].to_csv(processed_dir / "master_train.csv", index=False)
    out["master_test"].to_csv(processed_dir / "master_test.csv", index=False)
    out["weather"].to_csv(processed_dir / "weather_hourly_processed.csv", index=False)
    out["events"].assign(zones=out["events"].zones.str.join("|")).to_csv(
        processed_dir / "events_processed.csv", index=False)
    out["log"].frame().to_csv(reports_dir / "A1_cleaning_log.csv", index=False)


def run(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, reports_dir: Path = REPORTS_DIR) -> dict:
    out = build_master_tables(load_raw(raw_dir))
    assert validate(out), "integrity checks failed – nothing written"
    export(out, processed_dir, reports_dir)
    print(f"master_train {out['master_train'].shape} | master_test {out['master_test'].shape} -> {processed_dir}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: data/processed + reports)")
    a = ap.parse_args()
    run(processed_dir=a.out or PROCESSED_DIR, reports_dir=a.out or REPORTS_DIR)
