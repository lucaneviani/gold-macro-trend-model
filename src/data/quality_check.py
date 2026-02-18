"""
quality_check.py — Verifica qualità di tutti i dati del progetto Gold Trend Prediction Model

Esecuzione:
    cd gold_model
    python -m src.data.quality_check

Produce un report su:
  - Copertura date per ogni serie
  - % valori mancanti per periodo (pre/post 2008, pre/post 2020)
  - Outlier (valori oltre 4σ)
  - Allineamento minimo: tutte le serie coprono 2005-2020?
  - Parsing e normalizzazione dei file manuali (GPR, WGC, COT)
"""

import sys
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
    RAW_FRED_DIR, RAW_YFINANCE_DIR, RAW_MANUAL_DIR,
    PROCESSED_DIR, FRED_SERIES, YFINANCE_TICKERS,
)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
COVERAGE_START = pd.Timestamp("2005-01-01")
COVERAGE_END   = pd.Timestamp("2024-01-01")  # Minimo accettabile
OUTLIER_SIGMA  = 4.0
PERIODS = {
    "pre-2008":   (pd.Timestamp("2004-01-01"), pd.Timestamp("2008-01-01")),
    "2008-2020":  (pd.Timestamp("2008-01-01"), pd.Timestamp("2020-01-01")),
    "2020-oggi":  (pd.Timestamp("2020-01-01"), pd.Timestamp("2030-01-01")),
}


# ===========================================================================
# PARSING FILE MANUALI
# ===========================================================================

def parse_gpr(path: Path) -> pd.DataFrame:
    """
    Legge gpr.xls e restituisce serie giornaliera con colonne:
      GPRD, GPRD_ACT, GPRD_THREAT, GPRD_MA30, GPRD_MA7
    Il file è in formato "long" con una riga per data × variabile.
    Filtriamo le righe dove la data è effettiva (colonna 'date' valida)
    e prendiamo la prima colonna numerica per ognuna.
    """
    df_raw = pd.read_excel(path, sheet_name="Sheet1")
    df_raw.columns = [c.strip() for c in df_raw.columns]

    # Il file ha colonne: DAY, N10D, GPRD, GPRD_ACT, GPRD_THREAT, date, GPRD_MA30, GPRD_MA7, ...
    # Ogni riga è una osservazione giornaliera
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
    df_raw = df_raw.dropna(subset=["date"])

    # Tieni solo le colonne numeriche utili
    keep = ["date", "GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA30", "GPRD_MA7"]
    keep = [c for c in keep if c in df_raw.columns]
    df = df_raw[keep].copy()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def parse_wgc_demand(path: Path) -> pd.DataFrame:
    """
    Legge wgc_demand.xlsx foglio 'Gold Balance' (dati annuali in formato wide).
    Traspone per ottenere un DataFrame con anno come indice e categorie come colonne.
    Restituisce serie annuale forward-fillable.
    """
    df_raw = pd.read_excel(path, sheet_name="Gold Balance", header=None)

    # Trova la riga che contiene gli anni (first non-NaN values are years 2000-2030)
    # Struttura: col 0 = NaN (padding), col 1 = etichetta, col 2+ = dati
    header_row = None
    label_col  = None
    data_start_col = None

    for i in range(len(df_raw)):
        # Cerca la colonna da cui iniziano i dati numerici (anni)
        for col_offset in range(0, min(5, df_raw.shape[1] - 3)):
            val = df_raw.iloc[i, col_offset + 2]
            if not pd.isna(val):
                try:
                    year = int(float(val))
                    if 2000 <= year <= 2030:
                        # Verifica che anche le colonne successive siano anni consecutivi
                        next_val = df_raw.iloc[i, col_offset + 3]
                        if not pd.isna(next_val) and int(float(next_val)) == year + 1:
                            header_row = i
                            label_col = col_offset + 1   # colonna etichette (tipicamente 1)
                            data_start_col = col_offset + 2
                            break
                except (ValueError, TypeError):
                    pass
        if header_row is not None:
            break

    if header_row is None:
        raise ValueError("Non trovata la riga degli anni nel foglio Gold Balance")

    years_series = df_raw.iloc[header_row, data_start_col:].dropna()
    years = []
    for y in years_series:
        try:
            yr = int(float(y))
            if 2000 <= yr <= 2030:
                years.append(yr)
        except (ValueError, TypeError):
            pass  # Ignora celle non-numeriche o non-anno a destra dello sheet

    # Estrai righe dati (quelle con etichetta di categoria nella colonna etichette)
    data = {}
    for i in range(header_row + 1, len(df_raw)):
        label = df_raw.iloc[i, label_col]
        if pd.isna(label) or not isinstance(label, str):
            continue
        label = label.strip()
        if not label or label.startswith("Data") or label.startswith("Source") or label.startswith("Note"):
            continue
        values = df_raw.iloc[i, data_start_col:data_start_col + len(years)].values
        if len(values) == len(years):
            data[label] = [pd.to_numeric(v, errors="coerce") for v in values]

    df = pd.DataFrame(data, index=years)
    df.index = pd.to_datetime([f"{y}-12-31" for y in df.index])
    df.index.name = "date"

    # Rinomina colonne chiave con nomi puliti
    rename_map = {}
    for col in df.columns:
        key = col.lower().strip().lstrip()
        if "total supply" in key:
            rename_map[col] = "wgc_total_supply"
        elif "total demand" in key:
            rename_map[col] = "wgc_total_demand"
        elif "jewellery fabrication" in key:
            rename_map[col] = "wgc_jewellery"
        elif "investment" in key and "bar" not in key and "etf" not in key.lower():
            rename_map[col] = "wgc_investment"
        elif "etf" in key.lower():
            rename_map[col] = "wgc_etf_flows"
        elif "central bank" in key:
            rename_map[col] = "wgc_central_bank_demand"
        elif "mine production" in key.lstrip():
            rename_map[col] = "wgc_mine_production"
        elif "lbma gold price" in key.lower():
            rename_map[col] = "wgc_lbma_price"
    df = df.rename(columns=rename_map)

    return df


def parse_cot(path: Path) -> pd.DataFrame:
    """Legge il file COT oro già salvato da download_data."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.sort_index()


# ===========================================================================
# FUNZIONI DI ANALISI QUALITÀ
# ===========================================================================

def analyze_series(name: str, series: pd.Series) -> dict:
    """Analizza una singola serie temporale e ritorna un dizionario di metriche."""
    result = {
        "name": name,
        "n_obs": len(series),
        "date_start": series.index.min().date() if len(series) > 0 else None,
        "date_end": series.index.max().date() if len(series) > 0 else None,
        "pct_missing_total": round(series.isna().mean() * 100, 2),
        "covers_target_window": False,
        "outliers_n": 0,
        "outlier_values": [],
    }

    if len(series) == 0:
        return result

    # Copertura finestra target
    result["covers_target_window"] = (
        series.index.min() <= COVERAGE_START and
        series.index.max() >= COVERAGE_END
    )

    # Valori mancanti per periodo
    for period_name, (p_start, p_end) in PERIODS.items():
        mask = (series.index >= p_start) & (series.index < p_end)
        sub = series[mask]
        pct = sub.isna().mean() * 100 if len(sub) > 0 else None
        result[f"missing_{period_name}"] = round(pct, 2) if pct is not None else "N/D"

    # Outlier (>4σ sul log-return o sul livello se non variazione)
    clean = series.dropna()
    if len(clean) > 30:
        mean, std = clean.mean(), clean.std()
        if std > 0:
            z_scores = (clean - mean) / std
            outlier_mask = z_scores.abs() > OUTLIER_SIGMA
            result["outliers_n"] = int(outlier_mask.sum())
            if outlier_mask.sum() > 0:
                outs = clean[outlier_mask]
                result["outlier_values"] = [
                    f"{d.date()}: {v:.2f} (z={z:.1f})"
                    for d, v, z in zip(outs.index[:5], outs.values[:5], z_scores[outlier_mask].values[:5])
                ]

    return result


def check_source(label: str, df: pd.DataFrame) -> list:
    """Analizza tutte le colonne numeriche di un DataFrame."""
    results = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].copy()
        series.index = pd.to_datetime(series.index)
        r = analyze_series(f"{label}.{col}", series)
        results.append(r)
    return results


# ===========================================================================
# STAMPA REPORT
# ===========================================================================

def print_section(title: str):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def print_series_report(results: list):
    header = f"  {'SERIE':<35} {'RIGHE':>6}  {'INIZIO':<12} {'FINE':<12} {'MISS%':>6}  {'COPRE?':<7}  {'OUTLIER':>7}"
    print(header)
    print("  " + "-" * 90)
    for r in results:
        copre = "✓" if r["covers_target_window"] else "✗ WARN"
        out_n = r["outliers_n"]
        out_str = f"{out_n}" if out_n == 0 else f"⚠ {out_n}"
        print(
            f"  {r['name']:<35} {r['n_obs']:>6}  "
            f"{str(r['date_start']):<12} {str(r['date_end']):<12} "
            f"{r['pct_missing_total']:>6.1f}%  {copre:<7}  {out_str:>7}"
        )
    # Missing per periodo
    print()
    header2 = f"  {'SERIE':<35} {'pre-2008':>9}  {'2008-2020':>10}  {'2020-oggi':>10}"
    print(header2)
    print("  " + "-" * 70)
    for r in results:
        p1 = str(r.get("missing_pre-2008", "N/D"))
        p2 = str(r.get("missing_2008-2020", "N/D"))
        p3 = str(r.get("missing_2020-oggi", "N/D"))
        print(f"  {r['name']:<35} {p1:>8}%  {p2:>9}%  {p3:>9}%")


def print_outlier_detail(results: list):
    any_out = False
    for r in results:
        if r["outliers_n"] > 0:
            any_out = True
            print(f"\n  {r['name']} — {r['outliers_n']} outlier >4σ:")
            for v in r["outlier_values"]:
                print(f"    {v}")
    if not any_out:
        print("  Nessun outlier significativo rilevato.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 70)
    print("  QUALITY CHECK — Gold Trend Prediction Model")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = []
    parse_errors = []

    # -------------------------------------------------------------------
    # 1. SERIE FRED
    # -------------------------------------------------------------------
    print_section("1. SERIE FRED")
    fred_results = []
    for name in FRED_SERIES:
        csv_path = RAW_FRED_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"  ⚠ File non trovato: {csv_path.name}")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        res_list = check_source(name, df)
        fred_results.extend(res_list)

    print_series_report(fred_results)
    all_results.extend(fred_results)

    # -------------------------------------------------------------------
    # 2. SERIE YFINANCE
    # -------------------------------------------------------------------
    print_section("2. SERIE YFINANCE")
    yf_results = []
    for name in YFINANCE_TICKERS:
        csv_path = RAW_YFINANCE_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"  ⚠ File non trovato: {csv_path.name}")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Per il prezzo principale usa solo 'close'
        if "close" in df.columns:
            res = check_source(name, df[["close"]])
        else:
            res = check_source(name, df)
        yf_results.extend(res)

    print_series_report(yf_results)
    all_results.extend(yf_results)

    # -------------------------------------------------------------------
    # 3. FILE MANUALI
    # -------------------------------------------------------------------
    print_section("3. FILE MANUALI")

    manual_results = []

    # GPR
    gpr_path = RAW_MANUAL_DIR / "gpr.xls"
    if gpr_path.exists():
        try:
            gpr_df = parse_gpr(gpr_path)
            r = check_source("GPR", gpr_df[["GPRD"]])
            manual_results.extend(r)
            print(f"  GPR: {len(gpr_df)} righe | "
                  f"{gpr_df.index.min().date()} → {gpr_df.index.max().date()} | "
                  f"Colonne: {list(gpr_df.columns)}")
        except Exception as e:
            parse_errors.append(f"GPR: {e}")
            print(f"  ⚠ Errore parsing GPR: {e}")
    else:
        print("  ⚠ gpr.xls non trovato")

    # WGC Demand
    wgc_path = RAW_MANUAL_DIR / "wgc_demand.xlsx"
    if wgc_path.exists():
        try:
            wgc_df = parse_wgc_demand(wgc_path)
            print(f"\n  WGC Demand (Gold Balance): {len(wgc_df)} anni annuali | "
                  f"{wgc_df.index.min().year} → {wgc_df.index.max().year}")
            print(f"  Colonne estratte: {list(wgc_df.columns)}")
            print(f"  (Nota: dati annuali → verranno forward-filled a frequenza settimanale nello Step 4)")
            r = check_source("WGC_DEMAND", wgc_df)
            manual_results.extend(r)
        except Exception as e:
            parse_errors.append(f"WGC Demand: {e}")
            print(f"  ⚠ Errore parsing WGC Demand: {e}")
    else:
        print("  ⚠ wgc_demand.xlsx non trovato")

    # WGC Central Banks (solo nota — non è serie storica)
    wcb_path = RAW_MANUAL_DIR / "wcg_central_banks.xlsx"
    if wcb_path.exists():
        print(f"\n  WGC Central Banks (wcg_central_banks.xlsx):")
        print(f"  ⚠ ATTENZIONE: file contiene uno SNAPSHOT statico (febbraio 2026)")
        print(f"     Non è una serie storica temporale → NON verrà usato nel modello.")
        print(f"     Per avere storico acquisti banche centrali serve IMF IFS o API gold.org.")

    # COT
    cot_path = RAW_MANUAL_DIR / "cot_gold.csv"
    if cot_path.exists():
        try:
            cot_df = parse_cot(cot_path)
            r = check_source("COT", cot_df[["nc_long", "nc_short", "open_interest"]])
            manual_results.extend(r)
            print(f"\n  COT Gold: {len(cot_df)} righe settimanali | "
                  f"{cot_df.index.min().date()} → {cot_df.index.max().date()}")
            print(f"  Colonne disponibili: {list(cot_df.columns)}")
        except Exception as e:
            parse_errors.append(f"COT: {e}")
            print(f"  ⚠ Errore parsing COT: {e}")
    else:
        print("  ⚠ cot_gold.csv non trovato")

    if manual_results:
        print()
        print_series_report(manual_results)
    all_results.extend(manual_results)

    # -------------------------------------------------------------------
    # 4. OUTLIER DETAIL
    # -------------------------------------------------------------------
    print_section("4. DETTAGLIO OUTLIER (>4σ)")
    print_outlier_detail(all_results)

    # -------------------------------------------------------------------
    # 5. RIEPILOGO COPERTURA
    # -------------------------------------------------------------------
    print_section("5. RIEPILOGO COPERTURA WINDOW TARGET (2005-2024)")
    ok = [r for r in all_results if r["covers_target_window"]]
    ko = [r for r in all_results if not r["covers_target_window"]]
    print(f"\n  Serie che coprono la finestra: {len(ok)}/{len(all_results)}")
    if ko:
        print(f"\n  Serie con COPERTURA INSUFFICIENTE:")
        for r in ko:
            print(f"    ✗ {r['name']}: {r['date_start']} → {r['date_end']}")

    # -------------------------------------------------------------------
    # 6. ERRORI DI PARSING
    # -------------------------------------------------------------------
    if parse_errors:
        print_section("6. ERRORI DI PARSING")
        for e in parse_errors:
            print(f"  ✗ {e}")

    # -------------------------------------------------------------------
    # 7. SALVA OUTPUT NORMALIZZATO
    # -------------------------------------------------------------------
    print_section("7. SALVATAGGIO FILE NORMALIZZATI")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # GPR normalizzato
    if (RAW_MANUAL_DIR / "gpr.xls").exists():
        try:
            gpr_df = parse_gpr(RAW_MANUAL_DIR / "gpr.xls")
            gpr_df.to_csv(RAW_MANUAL_DIR / "gpr_parsed.csv")
            print(f"  ✓ gpr_parsed.csv salvato ({len(gpr_df)} righe giornaliere)")
        except Exception as e:
            print(f"  ✗ gpr_parsed.csv — errore: {e}")

    # WGC demand normalizzato
    if (RAW_MANUAL_DIR / "wgc_demand.xlsx").exists():
        try:
            wgc_df = parse_wgc_demand(RAW_MANUAL_DIR / "wgc_demand.xlsx")
            wgc_df.to_csv(RAW_MANUAL_DIR / "wgc_demand_parsed.csv")
            print(f"  ✓ wgc_demand_parsed.csv salvato ({len(wgc_df)} righe annuali)")
        except Exception as e:
            print(f"  ✗ wgc_demand_parsed.csv — errore: {e}")

    print()
    print("=" * 70)
    print("  QUALITY CHECK COMPLETATO")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
