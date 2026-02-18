"""
config.py — Parametri globali del progetto Gold Trend Prediction Model

ISTRUZIONI:
  Imposta FRED_API_KEY con la tua chiave gratuita ottenuta su:
  https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Carica variabili d'ambiente da .env (se presente)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv opzionale — in alternativa usa variabili d'ambiente di sistema

# ---------------------------------------------------------------------------
# CHIAVE API FRED
# ---------------------------------------------------------------------------
# Crea un file .env nella root del progetto con:
#   FRED_API_KEY=your_key_here
# Chiave gratuita: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# PARAMETRI TEMPORALI
# ---------------------------------------------------------------------------
START_DATE = "2004-01-01"   # Inizio storico (2004 per avere ~20 anni di dati)
END_DATE   = None           # None = usa la data odierna

# Frequenza settimanale — venerdì come giorno di riferimento
WEEKLY_FREQ = "W-FRI"

# ---------------------------------------------------------------------------
# PARAMETRI DEL TARGET
# ---------------------------------------------------------------------------
TARGET_HORIZONS_WEEKS = [12, 16, 26]  # Orizzonti predittivi (settimane) — medio-lungo termine
TARGET_THRESHOLD_PCT  = 0.02          # Soglia ±2% per target binario

# Soglie dello score finale 0-100
SCORE_LONG_THRESHOLD  = 65   # Score > 65 → segnale rialzista
SCORE_SHORT_THRESHOLD = 35   # Score < 35 → segnale ribassista

# ---------------------------------------------------------------------------
# SERIE FRED
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # GOLD_LONDON (GOLDAMGBD228NLBM) rimossa: serie non più disponibile su FRED.
    # Prezzo oro di riferimento: GOLD_FUTURES (GC=F) da yfinance.
    "REAL_YIELD_10Y": "DFII10",           # TIPS 10Y real yield (giornaliero)
    "BREAKEVEN_10Y":  "T10YIE",           # Breakeven inflation 10Y (giornaliero)
    "BREAKEVEN_5Y":   "T5YIE",            # Breakeven inflation 5Y (giornaliero)
    "FED_FUNDS":      "FEDFUNDS",         # Fed Funds Rate (mensile)
    "CPI":            "CPIAUCSL",         # CPI inflazione USA (mensile)
    "VIX_FRED":       "VIXCLS",           # VIX (giornaliero)
    "FSI":            "STLFSI4",          # St. Louis Financial Stress Index (settimanale)
    "EPU":            "USEPUINDXD",       # Economic Policy Uncertainty (giornaliero)
    "DFF":            "DFF",              # Fed Funds effective rate (giornaliero)
}

# ---------------------------------------------------------------------------
# TICKER YFINANCE
# ---------------------------------------------------------------------------
YFINANCE_TICKERS = {
    "GOLD_FUTURES":   "GC=F",      # Futures oro continui
    "DXY":            "DX-Y.NYB",  # Dollar Index
    "GLD":            "GLD",       # ETF SPDR Gold Shares
    "IAU":            "IAU",       # ETF iShares Gold Trust
    "VIX_YF":         "^VIX",      # VIX (backup FRED)
    "TLT":            "TLT",       # ETF Treasury 20Y
    "TNX":            "^TNX",      # Yield Treasury 10Y
}

# ---------------------------------------------------------------------------
# DATI MANUALI IN /data/raw/manual/
# ---------------------------------------------------------------------------
MANUAL_FILES = {
    # GPR: scaricato manualmente da matteoiacoviello.com/gpr.htm
    "GPR":          "gpr.xls",
    # COT: scaricato automaticamente via libreria cot-reports (cot_gold.csv)
    "COT":          "cot_gold.csv",
    # WGC demand: scaricato manualmente da gold.org (annuale 2010-2025)
    "WGC_DEMAND":   "wgc_demand.xlsx",
    # WGC central banks: SCARTATO — solo snapshot statico, non serie storica
    # "WGC_CB":     "wcg_central_banks.xlsx",
    # File normalizzati (output di quality_check.py)
    "GPR_PARSED":       "gpr_parsed.csv",
    "WGC_DEMAND_PARSED": "wgc_demand_parsed.csv",
}

# ---------------------------------------------------------------------------
# PERCORSI DEL PROGETTO
# ---------------------------------------------------------------------------
# Radice del progetto (questo file si trova in gold_model/)
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR            = PROJECT_ROOT / "data"
RAW_DIR             = DATA_DIR / "raw"
RAW_FRED_DIR        = RAW_DIR / "fred"
RAW_YFINANCE_DIR    = RAW_DIR / "yfinance"
RAW_MANUAL_DIR      = RAW_DIR / "manual"
PROCESSED_DIR       = DATA_DIR / "processed"

SRC_DIR             = PROJECT_ROOT / "src"
NOTEBOOKS_DIR       = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR         = PROJECT_ROOT / "outputs"
CHARTS_DIR          = OUTPUTS_DIR / "charts"
RESULTS_DIR         = OUTPUTS_DIR / "results"

# File output principali
DATASET_RAW_PATH      = PROCESSED_DIR / "dataset_raw.csv"
DATASET_FEATURES_PATH = PROCESSED_DIR / "dataset_features.csv"

# ---------------------------------------------------------------------------
# PARAMETRI MODELLO
# ---------------------------------------------------------------------------
# Walk-forward validation
WF_TRAIN_START = "2005-01-01"
WF_FIRST_TEST  = "2016-01-01"   # Prima finestra di test

# LightGBM — parametri di default (possono essere sovrascritti da tuning)
LGBM_PARAMS_BASE = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "n_estimators":     300,
    "learning_rate":    0.03,
    "num_leaves":       15,       # Piccolo per evitare overfitting
    "max_depth":        4,
    "min_child_samples": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbose":          -1,
}

# Rolling window per analisi di stabilità (settimane)
ROLLING_CORR_WINDOW = 156   # ~3 anni di settimane
