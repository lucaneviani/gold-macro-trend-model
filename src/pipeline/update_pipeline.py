"""
update_pipeline.py — Pipeline di aggiornamento settimanale

Esecuzione (ogni settimana, il venerdì dopo la chiusura dei mercati):
    cd gold_model
    python -m src.pipeline.update_pipeline

OPPURE usare lo script di convenienza nella root:
    python run_weekly.py

COSA FA:
    1. Scarica dati aggiornati (FRED + yfinance) — idempotente
    2. Ricostruisce dataset_raw.csv e dataset_features.csv
    3. Carica i modelli e i calibratori salvati (NO retraining)
    4. Calcola le probabilità per l'ultima settimana disponibile
    5. Calibra con Platt scaling
    6. Compila lo score composito 0-100
    7. Stampa il report e salva current_score.csv + score_history.csv

NOTA: Il retraining completo (model.py + calibrate.py) va eseguito
      periodicamente (es. ogni 3-6 mesi) per aggiornare i pesi del modello
      con i nuovi dati storici. La pipeline settimanale usa i modelli
      già addestrati — è rapida (pochi secondi).
"""

import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup path
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    SCORE_LONG_THRESHOLD,
    SCORE_SHORT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
HISTORY_PATH = RESULTS_DIR / "score_history.csv"

HORIZON_LABEL = {"target_12w": "12w", "target_16w": "16w", "target_26w": "26w"}
TARGETS_ALL   = ["target_12w", "target_16w", "target_26w"]
WEIGHTS       = {"target_12w": 0.25, "target_16w": 0.50, "target_26w": 0.25}


# ===========================================================================
# STEP 1 — Download dati aggiornati
# ===========================================================================

def step_download() -> bool:
    """
    Esegue il download incrementale di FRED e yfinance.
    Idempotente: se i dati sono già aggiornati non fa nulla.
    Ritorna True se completato senza errori fatali.
    """
    print("  [1/5] Download dati aggiornati...")
    try:
        from src.data.download_data import download_all_fred, download_all_yfinance, download_cot
        download_all_fred()
        download_all_yfinance()
        # COT: download solo se necessario (dati settimanali CFTC)
        # download_cot()  # Decommentare se si vuole aggiornare il COT
        print("       ✓ Download completato")
        return True
    except Exception as e:
        print(f"       ⚠ Download parziale: {e}")
        print("       → Continuo con i dati esistenti")
        return False


# ===========================================================================
# STEP 2 — Ricostruzione dataset_raw.csv
# ===========================================================================

def step_build_dataset() -> bool:
    print("  [2/5] Ricostruzione dataset settimanale...")
    try:
        from src.data.build_dataset import build_dataset, main as build_main
        # Reindirizza stdout per sopprimere l'output dettagliato
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_main()
        print("       ✓ dataset_raw.csv aggiornato")
        return True
    except Exception as e:
        print(f"       ✗ ERRORE: {e}")
        return False


# ===========================================================================
# STEP 3 — Ricostruzione feature
# ===========================================================================

def step_build_features() -> bool:
    print("  [3/5] Ricostruzione feature engineering...")
    try:
        import io, contextlib
        from src.features.feature_engineering import main as fe_main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fe_main()
        print("       ✓ dataset_features.csv e dataset_selected.csv aggiornati")
        return True
    except Exception as e:
        print(f"       ✗ ERRORE: {e}")
        return False


# ===========================================================================
# STEP 4 — Caricamento dati + modelli + calibratori
# ===========================================================================

def load_models_and_data() -> tuple:
    """
    Carica:
        - dataset_selected.csv (solo ultima riga necessaria per predizione)
        - selected_features.txt
        - lgbm_4w/8w/12w.pkl
        - platt_4w/8w/12w.pkl
    Ritorna (df, features, models_dict, platts_dict)
    """
    print("  [4/5] Caricamento modelli e calibratori...")

    # Dataset
    ds_path = PROCESSED_DIR / "dataset_selected.csv"
    sf_path = PROCESSED_DIR / "selected_features.txt"
    if not ds_path.exists():
        raise FileNotFoundError(f"{ds_path} non trovato. Eseguire il setup completo prima.")

    df = pd.read_csv(ds_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)

    # Feature list
    if sf_path.exists():
        features = sf_path.read_text(encoding="utf-8").strip().splitlines()
        features = [f for f in features if f in df.columns]
    else:
        # Fallback: tutte le colonne non-target
        features = [c for c in df.columns
                    if not c.startswith("target_") and not c.startswith("gold_fwd_")]

    # Modelli e calibratori
    lgbm_models = {}
    platt_models = {}
    for target in TARGETS_ALL:
        h = HORIZON_LABEL[target]
        mdl_path = MODELS_DIR / f"lgbm_{h}.pkl"
        plt_path = MODELS_DIR / f"platt_{h}.pkl"
        if mdl_path.exists():
            with open(mdl_path, "rb") as f:
                lgbm_models[target] = pickle.load(f)
        if plt_path.exists():
            with open(plt_path, "rb") as f:
                platt_models[target] = pickle.load(f)

    loaded = list(lgbm_models.keys())
    print(f"       ✓ Modelli caricati: {[HORIZON_LABEL[t] for t in loaded]}")
    print(f"       ✓ Calibratori caricati: {[HORIZON_LABEL[t] for t in platt_models]}")
    print(f"       ✓ Dataset: {df.shape[0]} righe, ultima settimana: {df.index[-1].strftime('%Y-%m-%d')}")

    return df, features, lgbm_models, platt_models


# ===========================================================================
# STEP 5 — Calcolo score + report
# ===========================================================================

def compute_current_score(
    df: pd.DataFrame,
    features: list[str],
    lgbm_models: dict,
    platt_models: dict,
) -> dict:
    """
    Predice sulla riga più recente del dataset.
    Ritorna dict con tutte le informazioni per il report.
    """
    last_row = df.iloc[[-1]]
    last_date = last_row.index[0]
    X_last = last_row[features]

    result = {
        "date": last_date,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    probs_raw = {}
    probs_cal = {}

    for target in TARGETS_ALL:
        h = HORIZON_LABEL[target]
        if target not in lgbm_models:
            continue
        model = lgbm_models[target]
        platt = platt_models.get(target)

        prob_raw = float(model.predict_proba(X_last)[:, 1][0])
        probs_raw[target] = prob_raw

        if platt is not None:
            prob_c = float(platt.predict_proba(np.array([[prob_raw]]))[:, 1][0])
        else:
            prob_c = prob_raw
        probs_cal[target] = prob_c

        result[f"prob_raw_{h}"] = round(prob_raw, 4)
        result[f"prob_cal_{h}"] = round(prob_c, 4)

    # Score composito
    valid = [t for t in TARGETS_ALL if t in probs_cal]
    if not valid:
        composite = np.nan
    else:
        total_w = sum(WEIGHTS[t] for t in valid)
        composite = sum(WEIGHTS[t] * probs_cal[t] for t in valid) / total_w

    score = round(composite * 100, 1) if not np.isnan(composite) else np.nan

    if np.isnan(score):
        signal = "UNDEFINED"
    elif score >= SCORE_LONG_THRESHOLD:
        signal = "LONG"
    elif score <= SCORE_SHORT_THRESHOLD:
        signal = "SHORT"
    else:
        signal = "FLAT"

    result["score"]          = score
    result["signal"]         = signal
    result["composite_prob"] = round(composite, 4) if not np.isnan(composite) else np.nan

    return result


def print_report(res: dict, df: pd.DataFrame) -> None:
    """Stampa il report settimanale formattato."""
    date_str = pd.Timestamp(res["date"]).strftime("%Y-%m-%d")
    score    = res["score"]
    signal   = res["signal"]

    # Emoji/indicatore visivo del segnale
    signal_bar = {
        "LONG":      "███████████████████ LONG  ████████████████████",
        "SHORT":     "───────────── SHORT ─────────────────────────",
        "FLAT":      "         ·········· FLAT ··········           ",
        "UNDEFINED": "              ??? UNDEFINED ???               ",
    }.get(signal, signal)

    # Score bar visuale (50 caratteri)
    if not np.isnan(score):
        filled = int(round(score / 2))   # 0-50
        bar = "█" * filled + "░" * (50 - filled)
        bar_line = f"  [{bar}] {score:.1f}/100"
    else:
        bar_line = "  [N/A]"

    print()
    print("╔" + "═" * 60 + "╗")
    print("║  GOLD TREND PREDICTION MODEL — Segnale Settimanale" + " " * 9 + "║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Data riferimento:  {date_str:<39}║")
    print(f"║  Generato:          {res['generated_at']:<39}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  SCORE:   {score:>5.1f} / 100{' '*44}║")
    print(f"║  {bar_line[2:]:<59}║")
    print(f"║  SEGNALE: {signal:<50}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Probabilità calibrate:{' '*36}║")
    p12 = res.get("prob_cal_12w", np.nan)
    p16 = res.get("prob_cal_16w", np.nan)
    p26 = res.get("prob_cal_26w", np.nan)
    print(f"║    +12 settimane:  {p12:>6.1%}  (raw: {res.get('prob_raw_12w',np.nan):>6.1%})      ║")
    print(f"║    +16 settimane:  {p16:>6.1%}  (raw: {res.get('prob_raw_16w',np.nan):>6.1%})      ║")
    print(f"║    +26 settimane:  {p26:>6.1%}  (raw: {res.get('prob_raw_26w',np.nan):>6.1%})      ║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Soglie: LONG > {SCORE_LONG_THRESHOLD}   FLAT {SCORE_SHORT_THRESHOLD}–{SCORE_LONG_THRESHOLD}   SHORT < {SCORE_SHORT_THRESHOLD}            ║")
    print(f"║  Pesi: 12w=25%  16w=50%  26w=25%{' '*28}║")
    print("╚" + "═" * 60 + "╝")

    # Contesto storico (ultimi 10 prezzi oro)
    if "GOLD" in df.columns:
        gold = df["GOLD"].dropna().tail(10)
        last_price  = gold.iloc[-1]
        price_4w    = gold.iloc[-5] if len(gold) >= 5 else np.nan
        price_8w    = gold.iloc[-9] if len(gold) >= 9 else np.nan
        ret_4w  = (last_price / price_4w - 1) * 100 if not np.isnan(price_4w) else np.nan
        ret_8w  = (last_price / price_8w - 1) * 100 if not np.isnan(price_8w) else np.nan
        print(f"\n  Contesto prezzo oro (GOLD_FUTURES):")
        print(f"    Prezzo corrente:  ${last_price:>8,.2f}")
        if not np.isnan(ret_4w):
            print(f"    Rend. ultime 4w:  {ret_4w:>+6.2f}%")
        if not np.isnan(ret_8w):
            print(f"    Rend. ultime 8w:  {ret_8w:>+6.2f}%")

    # Interpretazione
    print(f"\n  Interpretazione:")
    if signal == "LONG":
        print(f"    Il modello prevede una probabilità elevata che l'oro")
        print(f"    salga di ≥2% nelle prossime 16 settimane.")
        print(f"    → Segnale rialzista su orizzonti 12-26 settimane")
    elif signal == "SHORT":
        print(f"    Il modello non prevede un rialzo significativo dell'oro")
        print(f"    nelle prossime 16 settimane.")
        print(f"    → Segnale ribassista / difensivo")
    else:  # FLAT / UNDEFINED
        print(f"    Le probabilità non superano le soglie direzionali.")
        print(f"    → Segnale neutro: preferire posizione flat o ridotta")


# ===========================================================================
# STORICO SEGNALI
# ===========================================================================

def update_history(res: dict) -> None:
    """Appende il segnale corrente al file storico score_history.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([res])
    row["date"] = pd.to_datetime(row["date"]).dt.strftime("%Y-%m-%d")

    if HISTORY_PATH.exists():
        hist = pd.read_csv(HISTORY_PATH)
        # Rimuovi eventuale riga con stessa data (idempotente)
        hist = hist[hist["date"] != row["date"].values[0]]
        hist = pd.concat([hist, row], ignore_index=True)
    else:
        hist = row

    hist.to_csv(HISTORY_PATH, index=False)


# ===========================================================================
# MAIN
# ===========================================================================

def main(skip_download: bool = False, skip_rebuild: bool = False) -> None:
    """
    Pipeline completa.

    Args:
        skip_download: se True salta il download (usa dati esistenti)
        skip_rebuild:  se True salta la ricostruzione dei dataset
    """
    print()
    print("=" * 64)
    print("  GOLD TREND PREDICTION MODEL — Pipeline Settimanale")
    print(f"  Avvio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    print()

    # Step 1: Download
    if not skip_download:
        step_download()
    else:
        print("  [1/5] Download saltato (--skip-download)")

    # Step 2: Ricostruzione dataset
    if not skip_rebuild:
        ok2 = step_build_dataset()
        if not ok2:
            print("  ATTENZIONE: errore nella ricostruzione del dataset. Uso dati esistenti.")
    else:
        print("  [2/5] Ricostruzione dataset saltata (--skip-rebuild)")

    # Step 3: Feature engineering
    if not skip_rebuild:
        ok3 = step_build_features()
        if not ok3:
            print("  ATTENZIONE: errore nel feature engineering. Uso feature esistenti.")
    else:
        print("  [3/5] Feature engineering saltato (--skip-rebuild)")

    # Step 4: Carica modelli e dati
    try:
        df, features, lgbm_models, platt_models = load_models_and_data()
    except FileNotFoundError as e:
        print(f"\n  ERRORE FATALE: {e}")
        print("  Eseguire il setup completo: src/models/model.py + src/models/calibrate.py")
        sys.exit(1)

    # Step 5: Calcolo score
    print("  [5/5] Calcolo score composito...")
    res = compute_current_score(df, features, lgbm_models, platt_models)
    print(f"       ✓ Score calc: {res['score']:.1f} | Segnale: {res['signal']}")

    # Report
    print_report(res, df)

    # Salvataggio
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Segnale corrente
    score_path = RESULTS_DIR / "current_score.csv"
    pd.DataFrame([res]).to_csv(score_path, index=False)

    # Storico
    update_history(res)

    print(f"\n  ✓ Salvato: {score_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Aggiornato: {HISTORY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\n  Pipeline completata: {datetime.now().strftime('%H:%M:%S')}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gold Trend — Pipeline Settimanale")
    parser.add_argument("--skip-download", action="store_true",
                        help="Salta il download dati (usa cache esistente)")
    parser.add_argument("--skip-rebuild",  action="store_true",
                        help="Salta ricostruzione dataset e feature")
    args = parser.parse_args()

    main(skip_download=args.skip_download, skip_rebuild=args.skip_rebuild)
