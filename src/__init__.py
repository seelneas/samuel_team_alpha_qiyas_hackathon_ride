"""Reusable pipeline for the Addis Ride Demand Forecasting project.

    python -m src.cleaning   # raw CSVs   -> data/processed/ (master tables, cleaned weather/events, dictionary, log)
    python -m src.train      # processed  -> models/final_model.joblib (+ validation report)
    python -m src.predict    # model      -> submission CSV

The notebooks remain the documented, step-by-step version of the same logic.
"""
