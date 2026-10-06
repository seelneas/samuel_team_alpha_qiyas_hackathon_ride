# Figure captions

**1. `fig01_gaps_and_missingness.png`** — Missing/invalid values per column in the three raw tables and a zone × day map of missing trip hours. So what: the gaps are structured — Ayat did not exist before 15 Mar and a 42-hour city-wide outage hit 12–13 May — so they are excluded rather than imputed as zero demand.

**2. `fig02_before_after_cleaning.png`** — Raw vs cleaned distributions of temperature, rain readings and trips. So what: a July block recorded in °F (60–77), −9999 rain sentinels and trip spikes far beyond driver capacity would each have distorted the model; cleaning removes them without changing the normal range.

**3. `fig03_demand_trend_with_holidays.png`** — Daily city-wide trips Jan–Oct with 7-day mean, linear trend and public holidays. So what: demand grows ~40% over ten months, so the model needs a trend feature or it will under-forecast November.

**4. `fig04_hour_by_weekday_heatmap.png`** — Mean trips by hour × weekday city-wide, for Kazanchis and for Bole. So what: one city-wide profile is not enough — business zones empty at weekends while Bole peaks on Friday/Saturday nights, so zone × hour × weekday interactions are essential.

**5. `fig05_zone_profiles.png`** — Weekday vs weekend hourly profile for the five zone types. So what: CBD/hub zones have 08:00 and 18:00 commuter peaks, residential zones peak at 07:00/19:00, Merkato is a midday plateau and Bole has airport and night peaks.

**6. `fig06_weather_timezone_check.png`** — Daily temperature curve on the raw vs corrected clock, and rain–demand correlation for each clock shift. So what: the weather file is in UTC; the +3 h correction puts the temperature peak at 15:00 and raises the rain–demand correlation from 0.06 to 0.42.

**7. `fig07_rain_effect.png`** — Demand ratio vs dry hours by rain class and zone type. So what: rain lifts demand roughly 1.15× (light) to 1.6× (heavy) in every zone type except Merkato, where outdoor shopping falls — the effect is sub-linear and zone-dependent.

**8. `fig08_event_study.png`** — Demand ratio from −6 h to +6 h around event start for four event types vs the non-event baseline. So what: football and concerts raise demand before and especially after the event, conferences slightly, while road closures suppress it — event windows must extend past the end time.

**9. `fig09_holiday_effects.png`** — City-wide daily trips on each public holiday relative to the same weekday 1–2 weeks around it. So what: most holidays cut demand by 5–26% (Good Friday, Genna, Eid al-Adha most), while Sunday holidays are near normal — a single holiday flag captures most of the effect.

**10. `fig10_model_comparison.png`** — Validation RMSE of every model with both baselines marked and rolling-origin error bars. So what: the optimised LightGBM (Poisson loss, 2–5-week lags, 8-week zone×weekday×hour profile) reaches RMSE 8.47 vs 11.30 for seasonal-naive (−25%) and 27.5 for the mean, and stays ahead on every rolling-origin fold (9.21 ± 0.71).

**11. `fig11_forecast_vs_actual.png`** — Forecast vs actual hourly trips for three contrasting zones over the validation fortnight. So what: the model follows daily peaks and weekend shifts closely in all three zones (MAE 5.85 trips ≈ 4.5 drivers per zone-hour); the largest misses are on the tallest event and rain spikes, which it under-shoots.

**12. `fig12_feature_importance.png`** — Permutation importance of the top features, with weather and event features highlighted. So what: the 8-week zone×weekday×hour profile dominates (it encodes the zone's normal hour), followed by the multi-week lag mean; rain is the strongest non-history feature, confirming the corrected weather join adds real signal, while individual event flags matter only in the few hours they cover.

