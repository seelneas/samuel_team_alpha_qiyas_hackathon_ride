# Addis Ride Demand Forecasting — TEAM ALPHA

Qiyas AI Hackathon. We forecast how many ride requests each zone of Addis Ababa will get, **hour by hour, for 1–14 November 2025**, so a ride-hailing company can place drivers before demand arrives.

**Live demo app:** https://samuelteamalphaqiyashackathonridegit-8r5qwklneappykxjvr5kc8c.streamlit.app/

**Team:** Nejat Akmel · Lydia Million · Amanuel Ayalew · Samuel Beshir · Selamawit Elias · Mihiretab

---

## Results at a glance

We tested the model on 18–31 October, two weeks it never saw during training.

| Method | RMSE (lower is better) | MAE |
|---|---|---|
| **Our final model** (LightGBM, average of 3 runs) | **8.47** | **5.85** |
| Simple rule: usual value for the same zone, weekday and hour | 11.30 | – |
| Predict the overall average | 27.54 | – |

- **MAE** is the average miss: our forecast is off by about **5.9 trips per zone-hour**, roughly 4–5 drivers.
- **RMSE** is similar, but large misses count more. It is the hackathon's main score.
- On four separate two-week test periods the model averaged **9.21 ± 0.71**, compared with 12.32 ± 1.83 for the simple rule. The improvement holds across different weeks.

## What we found

- **Daily rhythm:** demand peaks during morning and evening commutes on weekdays. Weekends have a later, flatter pattern.
- **Zones differ:** business areas (e.g. Bole, Kazanchis) follow office hours. Market areas like Merkato behave differently.
- **Rain increases demand** across the city: about +13% in light rain, +30% in moderate rain and +46% in heavy rain. Merkato is the exception, where rain *reduces* trips.
- **Holidays:** most public holidays have lower demand, by up to 26%.
- **Events:** big events such as football matches raise demand in nearby zones around the event time.
- **Weather data uses UTC time** (3 hours behind Addis). We proved this and shifted it to local time before joining.

## Project structure

```
data/raw/          original files from the organisers (unchanged)
data/processed/    cleaned and joined tables (master_train.csv, master_test.csv, …)
notebooks/         01 cleaning → 02 analysis → 03 visualisations → 04 modelling
src/               same logic as reusable scripts (cleaning, features, train, predict)
models/            saved final model
reports/           HTML reports, cleaning log, model scores, feature importance
figures/           12 charts and their captions (figure_captions.md)
docs/              project_documentation.pdf and the script that builds it
presentation/      team_alpha_slides.pptx
submission/        team_alpha_submission.csv  ← final forecast
app/               Streamlit demo for exploring forecasts by zone and date
```

## Where each deliverable lives

| Deliverable | Location |
|---|---|
| **A** Cleaning & integration | `notebooks/01_cleaning_and_integration.ipynb`, `reports/A_cleaning_and_integration.html`, `reports/A1_cleaning_log.csv`, `data/processed/master_train.csv`, `master_test.csv`, `data_dictionary_master.csv` |
| **B** Analysis report | `notebooks/02_analysis_report.ipynb`, `reports/B_analysis_report.html` |
| **C** Visualisations | `notebooks/03_visualizations.ipynb`, `figures/fig01…fig12*.png`, `figures/figure_captions.md` |
| **D** Modelling & evaluation | `notebooks/04_modeling_and_evaluation.ipynb`, `reports/D_model_evaluation.html`, `reports/model_results.csv`, `models/final_model.joblib` |
| **E** Demo app | `app/` and the live link above |
| **F** Slides | `presentation/team_alpha_slides.pptx` |
| **G** Structure & reproducibility | this README, `requirements.txt`, `submission/team_alpha_submission.csv` |

## How to run it

Use the project's virtual environment. Other Python installs, such as base Anaconda, are missing `lightgbm` and `pyarrow`, and the saved model cannot load without `pyarrow`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Re-create the submission:**
```bash
python -m src.cleaning   # raw data → data/processed/
python -m src.train      # trains and saves models/final_model.joblib
python -m src.predict    # writes submission/team_alpha_submission.csv
```
Or run the notebooks top to bottom, in this order (each one uses the previous one's outputs):
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_cleaning_and_integration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_analysis_report.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_visualizations.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_modeling_and_evaluation.ipynb
```
In VS Code or Jupyter, select `.venv` as the kernel and use "Run All".

**Demo app:** use the [live version](https://samuelteamalphaqiyashackathonridegit-8r5qwklneappykxjvr5kc8c.streamlit.app/) or run it locally:
```bash
streamlit run app/app.py
```

**Rebuild the documents:**
```bash
python docs/build_documentation.py      # docs/project_documentation.pdf
python presentation/build_slides.py     # presentation/team_alpha_slides.pptx
```

## How we kept the results honest

- **No peeking at the future.** Trip-count inputs only use data at least 14 days old, so every input is available when a 14-day forecast is made.
- **Time-based testing.** We always trained on earlier weeks and tested on later weeks, never a random split.
- **Forecast weather.** November predictions use the weather forecast file, not actual weather.
- **Reproducible.** Random seeds are fixed and package versions are pinned in `requirements.txt`. `python -m src.predict` reproduces the submission exactly, and `src.cleaning` writes the same files as notebook 01.

## Limitations

- The history has only 48 football matches and a few holidays, so effects of rare or unusual events are uncertain.
- Weather in the forecast file may differ from the real November weather. Rain-heavy days carry more uncertainty.
- The model assumes November demand patterns resemble September–October.

More detail is in `docs/project_documentation.pdf` and the HTML reports in `reports/`.