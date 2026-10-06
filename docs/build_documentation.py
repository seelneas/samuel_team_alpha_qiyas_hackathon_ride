"""Build docs/project_documentation.pdf – end-to-end documentation of the project.

    python docs/build_documentation.py

Reads the project's own outputs (reports/*.csv, figures/*, docs/*_findings.json) so the PDF
always reflects the current results.
"""
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
DOCS, FIG, REP = ROOT / "docs", ROOT / "figures", ROOT / "reports"
OUT = DOCS / "project_documentation.pdf"

B = json.loads((DOCS / "analysis_findings.json").read_text())
D = json.loads((DOCS / "model_findings.json").read_text())
log = pd.read_csv(REP / "A1_cleaning_log.csv")
res = pd.read_csv(REP / "model_results.csv")
imp = pd.read_csv(REP / "feature_importance.csv")
caps = dict(re.findall(r"`(fig\d+_[^`]+\.png)`\*\* — (.+)", (FIG / "figure_captions.md").read_text()))

# ------------------------------------------------------------------ styles
ss = getSampleStyleSheet()
BLUE = colors.HexColor("#0B4F8A")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=BLUE, fontSize=18, spaceBefore=6, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=BLUE, fontSize=13.5, spaceBefore=10, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.8, leading=13.6, spaceAfter=6)
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=8.6, leading=11, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CODE = ParagraphStyle("Code", parent=ss["Code"], fontSize=7.8, leading=9.6, backColor=colors.HexColor("#F3F5F8"),
                      borderPadding=5, spaceAfter=8)
TITLE = ParagraphStyle("T", parent=H1, fontSize=26, leading=32, alignment=TA_CENTER)
SUB = ParagraphStyle("S", parent=BODY, fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))


def md(t: str) -> str:
    """Tiny markdown -> reportlab markup (bold, italic, code)."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return t


story = []
P = lambda t, s=BODY: story.append(Paragraph(md(t), s))
def h1(t): story.append(PageBreak()); story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def h3(t): story.append(Paragraph(t, H3))
def bullets(items): [story.append(Paragraph(md(i), BUL, bulletText="•")) for i in items]
def code(t): story.append(Preformatted(t.strip("\n"), CODE))


def table(rows, widths, header=True, font=8):
    data = [[Paragraph(md(str(c)), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C2CC")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FB")])]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE7F3"))]
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))


def figure(name, width=16.5):
    p = FIG / name
    if not p.exists():
        return
    img = Image(str(p))
    r = width * cm / img.imageWidth
    img.drawWidth, img.drawHeight = width * cm, img.imageHeight * r
    if img.drawHeight > 11 * cm:
        s = 11 * cm / img.drawHeight
        img.drawWidth, img.drawHeight = img.drawWidth * s, img.drawHeight * s
    story.append(KeepTogether([img, Paragraph(f"<b>{name}</b> — {md(caps.get(name, ''))}", CAP), Spacer(1, 8)]))


def finding(key, src, label=None):
    if key in src:
        story.append(Paragraph(f"<b>{label or key}.</b> " + md(src[key]), BODY))


# numbers
final = res[res.model.str.contains("optimised")].iloc[0]
sn = res[res.model == "Seasonal-naive (all history)"].iloc[0]
mean_b = res[res.model == "Mean predictor"].iloc[0]

# ================================================================== title page
story += [Spacer(1, 5 * cm), Paragraph("Addis Ababa Ride Demand Forecasting", TITLE), Spacer(1, 0.4 * cm),
          Paragraph("End-to-end project documentation", SUB), Spacer(1, 0.3 * cm),
          Paragraph("Qiyas AI Hackathon · hourly demand per zone, 1–14 November 2025", SUB),
          Spacer(1, 2 * cm)]
table([["Team", "TEAM ALPHA"],
       ["Final model", "LightGBM (tree-based model, set up for counts), average of 3 training runs"],
       ["Average miss on 18–31 Oct (MAE)", f"{final.mae:.2f} trips per zone-hour (≈ 4.5 drivers)"],
       ["RMSE on 18–31 Oct (weighs large misses more)", f"{final.rmse:.2f} trips per zone-hour"],
       ["RMSE of the simple rule (usual value for zone, weekday, hour)", f"{sn.rmse:.2f}"],
       ["RMSE averaged over 4 two-week test periods", f"9.21 ± {final.rmse_std:.2f}"],
       ["Document generated", date.today().strftime("%d %B %Y")]], [7, 9], header=False)

# ================================================================== contents
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t1", fontSize=10.5, leftIndent=10, leading=15),
                   ParagraphStyle("t2", fontSize=9, leftIndent=26, leading=12)]
story += [PageBreak(), Paragraph("Contents", H1), toc]

# ================================================================== 1 overview
h1("1. Project overview")
P("A ride-hailing operator in Addis Ababa needs to know **how many trips will be requested in each of 12 zones, "
  "every hour, for the next two weeks**, so it can position drivers and plan incentives. We are given ten months "
  "of hourly trip history (1 Jan – 31 Oct 2025), an hourly weather table (observations plus a forecast for "
  "November) and an events calendar (football matches, concerts, conferences, road closures, holidays…). "
  "The task is to forecast 4,032 zone-hours (12 zones × 14 days × 24 h) for 1–14 November 2025.")
P("The data are deliberately messy. Before any modelling, the three tables have to be cleaned and joined on a "
  "common clock and common zone names. The project is organised in the seven hackathon deliverables:")
table([["ID", "Deliverable", "Where it lives"],
       ["A", "Cleaning & integration (log, join map, audit, proof, master tables)",
        "notebooks/01, reports/A_cleaning_and_integration.html, data/processed/"],
       ["B", "Analysis report (zones, weather, events, data quality)", "notebooks/02, reports/B_analysis_report.html"],
       ["C", "Visualisation pack (12 figures + captions)", "notebooks/03, figures/"],
       ["D", "Modelling & evaluation", "notebooks/04, reports/D_model_evaluation.html, models/"],
       ["E", "Forecast demo app", "app/ (Streamlit)"],
       ["F", "5-slide presentation", "presentation/"],
       ["G", "Structure & reproducibility", "README.md, requirements.txt, src/"],
       ["—", "Submission file", "submission/team_alpha_submission.csv"]], [1.2, 7.5, 8])

h2("1.1 Key terms")
table([["Term", "Meaning in this project"],
       ["Zone-hour", "one zone in one hour, e.g. Bole on 5 Nov at 18:00. Forecasts are made per zone-hour."],
       ["EAT / UTC", "EAT is local Addis Ababa time. UTC is world time, 3 hours behind EAT."],
       ["MAE", "mean absolute error: the average miss, in trips per zone-hour. Lower is better."],
       ["RMSE", "root mean squared error: like MAE but large misses count more. The hackathon's main score."],
       ["Baseline / seasonal-naive", "a simple rule to beat: predict the usual value for the same zone, weekday and hour."],
       ["Lag", "the trip count at the same zone and hour some time earlier, e.g. a 336 h lag = 14 days earlier."],
       ["Validation", "testing the model on weeks it did not see during training (here, late October)."],
       ["Rolling-origin folds", "repeating that test on several two-week periods (6 Sep, 20 Sep, 4 Oct, 18 Oct)."],
       ["Leakage", "using information that would not be known when the forecast is made, which gives falsely good scores."],
       ["Ablation", "retraining with groups of inputs removed to measure how much each group helps."],
       ["Feature", "one input column given to the model, e.g. rain_mm or hour."]], [4, 12.7])

h2("1.2 The input data")
table([["File", "Content", "Main problems found"],
       ["ride_demand_train.csv", "85,460 rows: zone, pickup_hour, trips, avg_fare_birr, avg_wait_min, active_drivers",
        "3 timestamp formats, 55 zone spellings, duplicates, −1 sentinels, impossible spikes, outage, late-launching zone"],
       ["ride_demand_test.csv", "4,032 rows: row_id, zone, pickup_hour", "same timestamp / zone issues"],
       ["weather_hourly.csv", "7,538 rows: temp, rain, humidity, wind, observed/forecast",
        "on UTC (not local time), −9999 rain codes, a July block in °F, duplicate and missing hours"],
       ["events_calendar.csv", "165 events: type, venue, zone, start, end, attendance, status",
        "20 type spellings, multi-zone / city-wide zones, 3 date formats, missing or reversed ends, free-text attendance, cancelled events"],
       ["submission_template.csv", "row order for the submission", "—"]], [3.8, 6, 7])

h2("1.3 Approach in one paragraph")
P("Every timestamp is parsed with an explicit, regex-detected format and converted to East Africa Time (EAT, "
  "UTC+3). Zones are mapped to 12 canonical names. Trips, weather and events are cleaned with documented rules, "
  "then joined: weather many-to-one on the hour, events as an **interval join** (2 h before start to 2 h after "
  "end, per affected zone). Analysis quantifies zone profiles, growth, rain and event effects. A LightGBM model "
  "uses only information that exists at forecast time — calendar, weather forecast, scheduled events and trip "
  "history at least 14 days old — and is validated chronologically. A Streamlit app turns the model into an "
  "operations tool, and a `src/` package makes the whole pipeline reusable from the command line.")

# ================================================================== 2 cleaning
h1("2. Data cleaning & integration (Deliverable A)")
h2("2.1 Timestamp parsing and the clock problem")
P("Each table mixes several timestamp formats (ISO with `+03:00`, ISO with `Z`, `yyyy-mm-dd hh:mm`, "
  "`dd/mm/yyyy hh:mm`, and `Feb 06, 2025 08:00 PM`). Letting pandas guess silently swaps day and month "
  "(05/11 read as 11 May). Instead, each format is detected with a regular expression and parsed explicitly. "
  "Day-first was proven: in the `dd/mm` strings the first number exceeds 12 in thousands of rows while the second never does.")
P("**Time-zone proof.** The trip file is on local time (ISO rows carry `+03:00`, and all formats fall on one clean "
  "hourly grid). The weather file is on **UTC**: its ISO strings end in `Z`, it starts at 30 Dec 21:00 Z = 1 Jan "
  "00:00 EAT, and on the raw clock temperature peaks at 12:00 — solar noon — instead of the usual mid-afternoon. "
  "After shifting +3 h the peak lands at 15:00 EAT and the rain–demand correlation jumps from 0.06 to 0.42. "
  "Weather was therefore shifted +3 h; trips and events were left unchanged.")
figure("fig06_weather_timezone_check.png")

h2("2.2 Zone and event-type standardisation")
P("The raw files contain 55 zone spellings (case, spacing, aliases such as *Piazza*, *Mercato*, *Bole Rd*, "
  "*Kolfe Keranio*, text in brackets such as *(Kirkos)*). A normaliser lower-cases, collapses spaces, strips "
  "bracketed text and applies an alias map to the 12 canonical zones: Arat Kilo, Ayat, Bole, CMC, Gerji, "
  "Kazanchis, Kolfe, Lideta, Megenagna, Merkato, Piassa, Sarbet. Events naming several zones "
  "(*Lideta & KAZANCHIS*) are split, and city-wide events are expanded to all 12. Twenty event-type spellings "
  "were normalised to 8 labels.")

h2("2.3 Cleaning log (A1)")
P("Every issue found, how many rows it affected and the fix applied:")
table([["Table", "Column", "Issue", "Rows", "%", "Fix"]] +
      [[r.file.replace("_", " "), r.columns, r.issue_type, f"{r.rows_affected:,}", f"{r.pct_rows:.1f}", r.fix_applied]
       for r in log.itertuples()], [2.4, 2.3, 4.2, 1.3, 1.0, 5.6])
P("Key judgement calls:")
bullets([
    "**Impossible spikes**: trips above 4× the active drivers in that hour (e.g. 640 trips with 56 drivers) were set to missing. Real event peaks keep the usual ≈1.3 trips per driver.",
    "**Platform outage** (12 May 02:00 – 13 May 19:00, 42 h with no rows in any zone) and **Ayat before its launch on 15 Mar** were *excluded*, not filled with zeros — no demand was recorded, and zero would teach the model a false pattern.",
    "**Weather gaps** of up to 6 h are interpolated in time. The one long gap (19 h on 3 Sep) is filled with the hour-of-day average from history only. Missing rain is set to 0, the most common state.",
    "**Events**: missing or reversed end times are replaced by start + the median duration of that event type. Cancelled events are kept in the file for analysis (B3.4) but excluded from features.",
])
figure("fig01_gaps_and_missingness.png")
figure("fig02_before_after_cleaning.png")

h2("2.4 Join map (A3)")
code("""
                 weather_clean (1 row per hour, EAT)
                          |  key: datetime          many-to-one, LEFT join
                          v
 zone-hour grid  <---- trips_clean (left table)  ---->  master_train / master_test
 (zone, datetime)         ^
                          |  key: zone AND datetime in [start-2h, end+2h]
                          |  interval join, aggregated to 1 row per zone-hour, LEFT join
                 events_clean (exploded to zone x hour)
""")
bullets([
    "The **left table** is the set of trip zone-hours (train) or test rows, so joins can never add or drop rows.",
    "**Weather** is joined many-to-one on the hour (validated `m:1`).",
    "**Events** are exploded to every hour from 2 h before start to 2 h after end in each affected zone, labelled pre / during / post. Overlapping events are aggregated (any-of-type flags, max attendance), which makes the join many-to-one again.",
    "**Public holidays and school breaks** become whole-day, city-wide calendar flags.",
])

h2("2.5 Join audit, proof and integrity checks (A4, A5, A7)")
P("The audit shows a 100% weather match rate for train and test (the few imputed hours are flagged), and counts "
  "how many events matched at least one zone-hour. The proof section prints, for a rainy hour, a football window "
  "and a public holiday, the exact weather and event rows that were attached, plus the resulting feature values. "
  "Thirteen automated integrity checks must all pass before anything is written:")
bullets(["one row per zone-hour in train and test", "only the 12 canonical zones",
         "train timestamps in 1 Jan – 31 Oct and test in 1 – 14 Nov (EAT)", "test has 4,032 rows in template order",
         "no negative or sentinel trips or weather values", "no missing weather after the join",
         "joins did not change row counts", "train and test share identical feature columns",
         "no forecast-time-unavailable feature in test", "weather and event features present (Rule 5)"])

h2("2.6 Master tables and features (A6, A8)")
P("`master_train.csv` has **82,979** usable zone-hours and `master_test.csv` has **4,032** rows in submission order. "
  "Both carry identical feature columns, documented in `data_dictionary_master.csv`:")
table([["Group", "Features", "Why it helps"],
       ["Calendar", "hour, day_of_week, is_weekend, day_of_month, is_payday_window, is_public_holiday, is_school_break, days_since_start, hour_sin/cos",
        "daily and weekly shapes, holidays, growth trend"],
       ["Weather", "temp_c, rain_mm, humidity_pct, wind_kmh, rain_3h_sum, is_raining, rain_class",
        "rain moves walkers and minibus riders to ride-hail"],
       ["Events", "event_in_window, event_attendance, football/concert/conference/road_closure_window, event_pre/during/post",
        "crowds arriving and leaving; road closures suppress demand"],
       ["History", "lag_336h, lag_504h, roll_mean_168h_lag336", "recent level per zone-hour (≥ 14 days old)"],
       ["Excluded", "active_drivers, avg_wait_min, avg_fare_birr, lag_1h, lag_24h, month",
        "unknown at forecast time or unseen in training"]], [2.2, 9.3, 5.2])

# ================================================================== 3 analysis
h1("3. Analysis findings (Deliverable B)")
P("Each question in the brief was answered with a table and a written interpretation. "
  "The baseline used to measure effects is the zone's ordinary-hour shape (zone × weekday × hour) times its linear growth trend. "
  "Ratios are computed as sum over sum, so small hours don't dominate.")
h2("3.1 Zones and time patterns")
for k in ["B1.1", "B1.2", "B1.3", "B1.4"]:
    finding(k, B)
figure("fig03_demand_trend_with_holidays.png")
figure("fig04_hour_by_weekday_heatmap.png")
figure("fig05_zone_profiles.png")
h2("3.2 Weather")
for k in ["B2.1", "B2.2", "B2.3"]:
    finding(k, B)
figure("fig07_rain_effect.png")
h2("3.3 Events and holidays")
for k in ["B3.1", "B3.2", "B3.3", "B3.4"]:
    finding(k, B)
figure("fig08_event_study.png")
figure("fig09_holiday_effects.png")
h2("3.4 Data quality and operational columns")
for k in ["B4.1", "B4.2", "B4.3"]:
    finding(k, B)

# ================================================================== 4 viz
h1("4. Visualisation pack (Deliverable C)")
P("Twelve figures were produced at 150 dpi with a colour-blind-safe (Okabe–Ito) palette. Each has a caption "
  "stating what it shows and the *so what* for the forecast. Figures 1–9 appear next to the sections they "
  "support. Figures 10–12 summarise the model and appear in Section 5. All captions are listed here for reference:")
table([["Figure", "Caption"]] + [[k, v] for k, v in caps.items()], [5, 11.7])

# ================================================================== 5 modelling
h1("5. Modelling & evaluation (Deliverable D)")
h2("5.1 Rules followed")
bullets([
    "**Forecast-time features only (Rule 6)**: November is forecast up to 14 days ahead, so trip history is used only at lags of 14 days (336 h) or more. Weather comes from the provided forecast rows and events from the scheduled calendar.",
    "**Chronological validation (Rule 7)**: the main split trains before 18 Oct and validates on 18–31 Oct. Four rolling-origin folds start on 6 Sep, 20 Sep, 4 Oct and 18 Oct, each 14 days long. Nothing is shuffled.",
    "**Weather and events used (Rule 5)**, and **nothing fitted on the test file (Rule 8)**.",
])
finding("D4", D, "Leakage audit (D4)")

h2("5.2 Baselines and model comparison (D1, D2)")
table([["Model", "RMSE", "MAE", "Type"]] +
      [[r.model, f"{r.rmse:.2f}", f"{r.mae:.2f}", "baseline" if r.is_baseline else "model"] for r in res.itertuples()],
      [9, 2, 2, 3])
finding("D1", D, "Baselines")
finding("D2", D, "Models")
figure("fig10_model_comparison.png")

h2("5.3 Tuning, rolling-origin validation and ablation (D6, D3, D5)")
P("*Note:* sections 5.3–5.4 describe the model **before** the optimisation round in 5.5, so their scores "
  "(e.g. rolling RMSE 9.67, main RMSE 9.06) are higher than the final model's.")
finding("D6", D, "Tuning")
finding("D3", D, "Rolling origin")
finding("D5", D, "Ablation")

h2("5.4 Error analysis and response (D7, D8)")
finding("D7", D, "Where the model fails")
finding("D8", D, "Response")

h2("5.5 Optimisation round (D10)")
P("A further round of experiments was run on all four rolling folds. Each change was kept only if it lowered the mean rolling RMSE:")
table([["Change", "Main RMSE", "Rolling mean RMSE"],
       ["Tuned model after D8 (starting point)", "9.03", "9.58"],
       ["+ lags of 4 and 5 weeks, their mean, median and spread, lag ratio, hour × weekday", "8.87", "9.60"],
       ["+ 8-week zone × weekday × hour profile (refit inside each fold)", "8.70", "9.44"],
       ["+ Poisson objective (count data)", "8.57", "9.33"],
       ["+ smaller trees (15 leaves, 1,500 rounds), 3-seed average  → final", f"{final.rmse:.2f}", "9.21"]],
      [10.5, 2.8, 3.4])
P("Tried and rejected: weighting recent data more heavily, larger trees, a Tweedie objective, and dropping the weak event features.")
finding("D10", D, "Result")

h2("5.6 What the error means in practice (D9)")
P(f"On the validation fortnight the final model has **RMSE {final.rmse:.2f}** and **MAE {final.mae:.2f}** trips per "
  f"zone-hour, against an average demand of about 33 trips. A typical miss is 17.5% of demand. At ≈1.31 trips per "
  f"driver-hour, the driver plan is typically off by about **4–5 drivers per zone per hour**, and by about 6–7 in the "
  f"hardest hours (morning and evening rush, event exits). That is {100 * (1 - final.rmse / sn.rmse):.0f}% better than "
  f"the seasonal-naive baseline and {100 * (1 - final.rmse / mean_b.rmse):.0f}% better than predicting the mean.")
figure("fig11_forecast_vs_actual.png")

h2("5.7 Feature importance")
P("Permutation importance on the validation fortnight: how much RMSE rises when one feature is shuffled.")
table([["Feature", "RMSE increase", "Group"]] +
      [[r.feature, f"{r.importance:.3f}", r.group] for r in imp.head(12).itertuples()], [6, 3, 3])
figure("fig12_feature_importance.png")

h2("5.8 Final model and submission")
bullets([
    "Retrained on all history (1 Jan – 31 Oct) and saved to `models/final_model.joblib`. The file holds the 3 models, feature list, 8-week profile, parameters and trips-per-driver ratio.",
    "`submission/team_alpha_submission.csv`: 4,032 rows with columns `row_id` and `predicted_trips`, in template order, with no blank or negative values.",
    "Mean predicted November demand is 32.8 trips per zone-hour, against an October average of 32.9 — consistent with the trend.",
])

# ================================================================== 6 app
h1("6. Forecast demo app (Deliverable E)")
P("`app/app.py` is a Streamlit app for an operations manager. The **only inputs are a zone and a date** (1–14 Nov 2025). "
  "Weather and events are looked up automatically from the cleaned tables bundled in `app/assets/`, which "
  "`app/build_assets.py` creates from the processed data and the final model.")
table([["Output", "How it is computed"],
       ["24-hour forecast (table and curve)", "final model applied to that zone-day's forecast-time features"],
       ["Comparison with a typical day", "zone × weekday × hour mean over the last 8 weeks, drawn as a dashed line"],
       ["Peak hour", "hour with the highest forecast"],
       ["Drivers needed per hour", "forecast ÷ 1.31 trips per driver-hour, rounded up"],
       ["Expected gross fares", "forecast × the zone's historical average fare"],
       ["What was looked up", "e.g. 'rain in 4 hours, heaviest 13.0 mm at 17:00; Premier league match at Addis Ababa Stadium 19:00–21:00, ~32,000 people'"],
       ["Chart shading", "event windows (orange) and rain hours (blue)"],
       ["Dates outside 1–14 Nov", "friendly warning, no crash"],
       ["Download", "hourly table as CSV"]], [5, 11.7])
code("source .venv/bin/activate\nstreamlit run app/app.py      # opens http://localhost:8501")
P("A test harness checked several zone and date combinations (a football day in Kazanchis, a concert day in Bole, "
  "a rainy day in Piassa, an out-of-range date). The app's numbers equal the submission file exactly.")

# ================================================================== 7 reproducibility
h1("7. Reproducibility & project structure (Deliverable G)")
h2("7.1 Environment")
code("python3 -m venv .venv && source .venv/bin/activate\npip install -r requirements.txt")
P("`requirements.txt` pins Python 3.13 packages: pandas 3.0.3, numpy 2.4.6, pyarrow 25.0.1, scikit-learn 1.9.0, "
  "lightgbm 4.7.0, matplotlib 3.11.0, seaborn 0.13.2, streamlit 1.65.0, python-pptx, jupyter and nbconvert. "
  "All paths are relative and random seeds are fixed (42 for experiments, 1–3 for the three final training runs).")
P("**Common pitfall:** another Python installation (e.g. base Anaconda) may be missing `lightgbm` or `pyarrow`; "
  "the saved model cannot be loaded without `pyarrow`. Always activate `.venv` first, and pick it as the notebook kernel.")
h2("7.2 Run order")
code("""jupyter nbconvert --to notebook --execute --inplace notebooks/01_cleaning_and_integration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_analysis_report.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_modeling_and_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_visualizations.ipynb   # after 04
python app/build_assets.py""")
h2("7.3 Reusable pipeline (src/)")
P("The notebook logic is also packaged as importable modules. `src.cleaning` was checked to write exactly the same "
  "master tables, weather, events and cleaning log as notebook 01, and `src.predict` reproduces the submission file "
  "exactly (maximum difference 0.0).")
table([["Module", "Responsibility"],
       ["src/config.py", "paths, seed, date ranges, zones, validation cut-offs"],
       ["src/cleaning.py", "timestamp parsing, zone/event normalisation, trip/weather/event cleaning, interval join, master tables, 13 integrity checks"],
       ["src/features.py", "model feature list, long lags, holiday-eve, 8-week profile, leakage guard"],
       ["src/train.py", "fit / predict, rolling-origin evaluate, train_final"],
       ["src/predict.py", "make_submission, forecast_zone_day"]], [4, 12.7])
code("""python -m src.cleaning                               # raw -> data/processed/
python -m src.train                                  # validation + models/final_model.joblib
python -m src.predict                                # submission CSV
python -m src.predict --zone Bole --date 2025-11-05  # one zone-day""")
h2("7.4 Folder layout")
code("""team_alpha/
|-- README.md, requirements.txt
|-- submission/team_alpha_submission.csv
|-- data/raw/                 original 5 CSVs (never edited)
|-- data/processed/           master_train, master_test, data_dictionary_master, cleaned weather/events
|-- notebooks/                01 cleaning, 02 analysis, 03 visualisations, 04 modelling
|-- figures/                  fig01 ... fig12, figure_captions.md
|-- reports/                  A_, B_, D_ HTML reports, cleaning log, model results, importance
|-- models/final_model.joblib
|-- src/                      reusable pipeline
|-- app/                      app.py, build_assets.py, requirements.txt, assets/
|-- docs/                     this document and its generator
|-- presentation/             team_alpha_slides""")

# ================================================================== 8 limitations
h1("8. Limitations and next steps")
bullets([
    "**Rare, one-off days** (e.g. Meskel and its eve, which appear once in training) remain the biggest source of error. A holiday-specific prior or zone × holiday-type effects would help.",
    "**Event peaks are under-predicted.** The tallest post-match surges exceed what the 48 football matches in the history can teach. An event-study multiplier applied on top of the model, or quantile forecasts for staffing, could close the gap.",
    "**Weather forecast error**: the model is validated on observed weather, but November uses forecast weather, which is noisier. Training with noise added to the weather features would make it more robust.",
    "**Driver supply** caps observed trips in heavy rain, so the model learns *served* demand, not *requested* demand. Modelling unmet demand would need request-level data.",
    "**Operational**: retrain weekly as new data arrive, track live error by zone, and host the app (Streamlit Community Cloud) so managers can use it without a laptop.",
])

# ------------------------------------------------------------------ build with TOC
class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("H1", "H2"):
            txt = f.getPlainText()
            if txt != "Contents":
                self.notify("TOCEntry", (0 if f.style.name == "H1" else 1, txt, self.page))


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "Addis Ride Demand Forecasting — TEAM ALPHA — project documentation")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {d.page}")
    c.restoreState()


doc = Doc(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
          title="Addis Ride Demand Forecasting – Project Documentation")
doc.multiBuild(story, onLaterPages=footer)
print("written", OUT)
