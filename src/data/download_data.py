"""
download_data.py — Download dati FRED e yfinance per il Gold Trend Prediction Model

Utilizzo:
    cd gold_model
    python -m src.data.download_data

Comportamento:
    - Idempotente: se un file CSV esiste già ed è stato aggiornato entro MAX_AGE_DAYS,
      viene saltato. Usa --force per riscaricare tutto.
    - Se una serie fallisce, viene loggato l'errore e si continua con le altre.
    - Al termine stampa un report completo (successi, fallimenti, copertura date).

Flag da riga di comando:
    --force     Riscaricare tutti i dati anche se già presenti
    --fred-only Scarica solo le serie FRED
    --yf-only   Scarica solo i ticker yfinance
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Setup: aggiungi la root del progetto al path se necessario
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent   # gold_model/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FRED_API_KEY, START_DATE, END_DATE,
    FRED_SERIES, YFINANCE_TICKERS,
    RAW_FRED_DIR, RAW_YFINANCE_DIR, RAW_MANUAL_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
MAX_AGE_DAYS = 3      # Considera "aggiornato" un file scaricato negli ultimi N giorni
MIN_ROWS     = 10     # Soglia minima di righe per considerare valido un download


# ===========================================================================
# FUNZIONI DI SUPPORTO
# ===========================================================================

def file_is_fresh(csv_path: Path, max_age_days: int = MAX_AGE_DAYS) -> bool:
    """Ritorna True se il file esiste ed è stato modificato di recente."""
    if not csv_path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(csv_path.stat().st_mtime)
    return age < timedelta(days=max_age_days)


def save_series(df: pd.DataFrame, path: Path, label: str) -> dict:
    """
    Salva un DataFrame come CSV e ritorna un dizionario con le statistiche.
    Il DataFrame deve avere l'indice come data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    info = {
        "label":      label,
        "path":       str(path.relative_to(PROJECT_ROOT)),
        "rows":       len(df),
        "start":      str(df.index.min().date()) if len(df) > 0 else "N/A",
        "end":        str(df.index.max().date()) if len(df) > 0 else "N/A",
        "status":     "OK" if len(df) >= MIN_ROWS else "WARNING: poche righe",
        "error":      None,
    }
    return info


# ===========================================================================
# DOWNLOAD FRED
# ===========================================================================

def download_fred_series(name: str, fred_id: str, force: bool = False) -> dict:
    """
    Scarica una singola serie FRED e la salva come CSV.
    Ritorna un dict con le info sul risultato.
    """
    csv_path = RAW_FRED_DIR / f"{name}.csv"

    # Controlla se il file è già fresco
    if not force and file_is_fresh(csv_path):
        existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        log.info(f"[FRED] {name:20s} → già aggiornato, skip ({len(existing)} righe)")
        return {
            "label":  name,
            "path":   str(csv_path.relative_to(PROJECT_ROOT)),
            "rows":   len(existing),
            "start":  str(existing.index.min().date()),
            "end":    str(existing.index.max().date()),
            "status": "SKIP (già recente)",
            "error":  None,
        }

    try:
        from fredapi import Fred
        if not FRED_API_KEY:
            raise EnvironmentError(
                "FRED_API_KEY mancante. Crea un file .env con:\n"
                "  FRED_API_KEY=la_tua_chiave\n"
                "Chiave gratuita: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        fred = Fred(api_key=FRED_API_KEY)

        end = END_DATE if END_DATE else datetime.today().strftime("%Y-%m-%d")
        series = fred.get_series(fred_id, observation_start=START_DATE, observation_end=end)

        if series is None or len(series) == 0:
            raise ValueError("Serie vuota restituita da FRED")

        df = pd.DataFrame({"value": series})
        df.index.name = "date"
        df = df.dropna()

        info = save_series(df, csv_path, name)
        log.info(f"[FRED] {name:20s} → OK | {info['rows']:5d} righe | {info['start']} → {info['end']}")
        return info

    except Exception as e:
        log.error(f"[FRED] {name:20s} → ERRORE: {e}")
        return {
            "label":  name,
            "path":   str(csv_path.relative_to(PROJECT_ROOT)),
            "rows":   0,
            "start":  "N/A",
            "end":    "N/A",
            "status": "ERRORE",
            "error":  str(e),
        }


def download_all_fred(force: bool = False) -> list[dict]:
    """Scarica tutte le serie FRED definite in config.py."""
    log.info("=" * 60)
    log.info("DOWNLOAD SERIE FRED")
    log.info("=" * 60)
    results = []
    for name, fred_id in FRED_SERIES.items():
        result = download_fred_series(name, fred_id, force=force)
        results.append(result)
    return results


# ===========================================================================
# DOWNLOAD YFINANCE
# ===========================================================================

def download_yfinance_ticker(name: str, ticker: str, force: bool = False) -> dict:
    """
    Scarica un singolo ticker yfinance e lo salva come CSV.
    Scarica OHLCV giornaliero. Salva solo Adj Close + Volume.
    """
    csv_path = RAW_YFINANCE_DIR / f"{name}.csv"

    if not force and file_is_fresh(csv_path):
        existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        log.info(f"[YF]   {name:20s} → già aggiornato, skip ({len(existing)} righe)")
        return {
            "label":  name,
            "path":   str(csv_path.relative_to(PROJECT_ROOT)),
            "rows":   len(existing),
            "start":  str(existing.index.min().date()),
            "end":    str(existing.index.max().date()),
            "status": "SKIP (già recente)",
            "error":  None,
        }

    try:
        import yfinance as yf

        end = END_DATE if END_DATE else datetime.today().strftime("%Y-%m-%d")

        tkr = yf.Ticker(ticker)
        df = tkr.history(start=START_DATE, end=end, auto_adjust=True)

        if df is None or len(df) == 0:
            raise ValueError("Nessun dato restituito da yfinance")

        # Normalizza le colonne: minuscolo, senza spazi
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"

        # Mantieni solo le colonne rilevanti (se presenti)
        keep_cols = [c for c in ["close", "volume", "open", "high", "low"] if c in df.columns]
        df = df[keep_cols]

        info = save_series(df, csv_path, name)
        log.info(f"[YF]   {name:20s} → OK | {info['rows']:5d} righe | {info['start']} → {info['end']}")
        return info

    except Exception as e:
        log.error(f"[YF]   {name:20s} → ERRORE: {e}")
        return {
            "label":  name,
            "path":   str(csv_path.relative_to(PROJECT_ROOT)),
            "rows":   0,
            "start":  "N/A",
            "end":    "N/A",
            "status": "ERRORE",
            "error":  str(e),
        }


def download_all_yfinance(force: bool = False) -> list[dict]:
    """Scarica tutti i ticker yfinance definiti in config.py."""
    log.info("=" * 60)
    log.info("DOWNLOAD TICKER YFINANCE")
    log.info("=" * 60)
    results = []
    for name, ticker in YFINANCE_TICKERS.items():
        result = download_yfinance_ticker(name, ticker, force=force)
        results.append(result)
    return results


# ===========================================================================
# DOWNLOAD COT (via cot-reports)
# ===========================================================================

def download_cot(force: bool = False) -> dict:
    """
    Scarica i dati COT oro COMEX usando la libreria cot-reports.
    Combina storico (cot_hist, 1986-2016) + anni recenti (cot_year, 2017-oggi).
    Filtra per GOLD - COMMODITY EXCHANGE INC. (codice commodity 88).
    Salva il risultato in /data/raw/manual/cot_gold.csv.
    """
    import sys as _sys
    log.info("=" * 60)
    log.info("DOWNLOAD COT GOLD (via cot-reports)")
    log.info("=" * 60)

    csv_path = RAW_MANUAL_DIR / "cot_gold.csv"

    if not force and file_is_fresh(csv_path):
        existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        log.info(f"[COT]  cot_gold.csv → già aggiornato, skip ({len(existing)} righe)")
        return {
            "label":  "COT_GOLD",
            "path":   str(csv_path.relative_to(PROJECT_ROOT)),
            "rows":   len(existing),
            "start":  str(existing.index.min().date()),
            "end":    str(existing.index.max().date()),
            "status": "SKIP (già recente)",
            "error":  None,
        }

    try:
        import cot_reports as cot
        import warnings
        warnings.filterwarnings("ignore")

        log.info("[COT]  Scarico storico 1986-2016...")
        df_hist = cot.cot_hist(cot_report_type="legacy_fut", store_txt=False, verbose=False)
        frames = [df_hist]

        current_year = datetime.now().year
        for yr in range(2017, current_year + 1):
            try:
                df_yr = cot.cot_year(year=yr, cot_report_type="legacy_fut", store_txt=False, verbose=False)
                frames.append(df_yr)
                _sys.stdout.write(f"\r[COT]  Scaricato anno {yr}   ")
                _sys.stdout.flush()
            except Exception:
                pass  # Anno corrente non ancora disponibile è normale

        print()
        df_all = pd.concat(frames, ignore_index=True)

        # Filtra oro COMEX
        gold = df_all[
            (df_all["CFTC Commodity Code"] == 88) &
            (df_all["Market and Exchange Names"].str.upper().str.contains("COMMODITY EXCHANGE"))
        ].copy()

        gold = gold.rename(columns={
            "As of Date in Form YYYY-MM-DD":        "date",
            "Market and Exchange Names":            "market",
            "Open Interest (All)":                  "open_interest",
            "Noncommercial Positions-Long (All)":   "nc_long",
            "Noncommercial Positions-Short (All)":  "nc_short",
            "Noncommercial Positions-Spreading (All)": "nc_spread",
            "Commercial Positions-Long (All)":      "comm_long",
            "Commercial Positions-Short (All)":     "comm_short",
            "Nonreportable Positions-Long (All)":   "nr_long",
            "Nonreportable Positions-Short (All)":  "nr_short",
        })

        gold["date"] = pd.to_datetime(gold["date"])
        gold = gold.sort_values("date").drop_duplicates("date")

        keep = ["date", "market", "open_interest", "nc_long", "nc_short", "nc_spread",
                "comm_long", "comm_short", "nr_long", "nr_short"]
        keep = [c for c in keep if c in gold.columns]
        gold = gold[keep].set_index("date")

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        gold.to_csv(csv_path)

        info = {
            "label":  "COT_GOLD",
            "path":   str(csv_path.relative_to(PROJECT_ROOT)),
            "rows":   len(gold),
            "start":  str(gold.index.min().date()),
            "end":    str(gold.index.max().date()),
            "status": "OK",
            "error":  None,
        }
        log.info(f"[COT]  cot_gold.csv → OK | {info['rows']} righe | {info['start']} → {info['end']}")
        return info

    except ImportError:
        err = "cot-reports non installato. Eseguire: pip install cot-reports"
        log.error(f"[COT]  {err}")
        return {"label": "COT_GOLD", "path": str(csv_path), "rows": 0,
                "start": "N/A", "end": "N/A", "status": "ERRORE", "error": err}
    except Exception as e:
        log.error(f"[COT]  ERRORE: {e}")
        return {"label": "COT_GOLD", "path": str(csv_path), "rows": 0,
                "start": "N/A", "end": "N/A", "status": "ERRORE", "error": str(e)}




def print_report(fred_results: list[dict], yf_results: list[dict]) -> None:
    """Stampa un report leggibile di tutti i download."""
    all_results = fred_results + yf_results

    ok      = [r for r in all_results if r["status"].startswith("OK") or r["status"].startswith("SKIP")]
    errors  = [r for r in all_results if r["status"] == "ERRORE"]
    warnings = [r for r in all_results if "WARNING" in r["status"]]

    print("\n")
    print("=" * 70)
    print("  REPORT DOWNLOAD — Gold Trend Prediction Model")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n{'SERIE/TICKER':<22} {'STATO':<28} {'RIGHE':>6}  {'DA':<12} {'A':<12}")
    print("-" * 84)
    for r in all_results:
        stato = r["status"] if r["error"] is None else f"ERRORE: {r['error'][:35]}"
        print(f"  {r['label']:<20} {stato:<28} {r['rows']:>6}  {r['start']:<12} {r['end']:<12}")

    print("-" * 84)
    print(f"\n  Successi / Skip:   {len(ok)}")
    print(f"  Errori:            {len(errors)}")
    print(f"  Warning:           {len(warnings)}")

    if errors:
        print("\n  SERIE CON ERRORI (richiedono attenzione):")
        for r in errors:
            print(f"    - {r['label']}: {r['error']}")

    if warnings:
        print("\n  SERIE CON WARNING (poche righe):")
        for r in warnings:
            print(f"    - {r['label']}: {r['rows']} righe")

    print("\n" + "=" * 70)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download dati FRED e yfinance per Gold Trend Prediction Model"
    )
    parser.add_argument("--force",     action="store_true", help="Riscaricare tutto anche se già presente")
    parser.add_argument("--fred-only", action="store_true", help="Scarica solo serie FRED")
    parser.add_argument("--yf-only",   action="store_true", help="Scarica solo ticker yfinance")
    parser.add_argument("--cot-only",  action="store_true", help="Scarica solo dati COT")
    args = parser.parse_args()

    fred_results = []
    yf_results   = []
    cot_results  = []

    if not args.yf_only and not args.cot_only:
        fred_results = download_all_fred(force=args.force)

    if not args.fred_only and not args.cot_only:
        yf_results = download_all_yfinance(force=args.force)

    if not args.fred_only and not args.yf_only:
        cot_result = download_cot(force=args.force)
        cot_results = [cot_result]

    print_report(fred_results, yf_results + cot_results)


if __name__ == "__main__":
    main()
