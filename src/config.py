"""Shared paths and constants (relative to the project root)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "submission"

SEED = 42
TZ_LOCAL = "Africa/Addis_Ababa"
TRAIN_START, TRAIN_END = pd.Timestamp("2025-01-01 00:00"), pd.Timestamp("2025-10-31 23:00")
TEST_START, TEST_END = pd.Timestamp("2025-11-01 00:00"), pd.Timestamp("2025-11-14 23:00")

CANONICAL_ZONES = ["Arat Kilo", "Ayat", "Bole", "CMC", "Gerji", "Kazanchis",
                   "Kolfe", "Lideta", "Megenagna", "Merkato", "Piassa", "Sarbet"]

# validation design (notebook 04)
VAL_START = pd.Timestamp("2025-10-18")
ROLLING_CUTS = [pd.Timestamp(d) for d in ("2025-09-06", "2025-09-20", "2025-10-04", "2025-10-18")]
HORIZON_DAYS = 14
