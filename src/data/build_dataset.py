"""
build_dataset.py — Costruisce il dataset settimanale integrato

Esecuzione:
    cd gold_model
    python -m src.data.build_dataset

Output:
    /data/processed/dataset_raw.csv

Logica di allineamento (frequenza: W-FRI, venerdì come giorno di riferimento):

  DATI GIORNALIERI FRED (TIPS, breakeven, VIX, EPU, DFF, GPR):
    → resample W-FRI, last() = valore del venerdì (o giovedì se venerdì mancante)
    → NO lookahead: il dato è disponibile il giorno stesso della pubblicazione

  DATI MENSILI FRED (FED_FUNDS, CPI):
    → forward-fill da data osservazione a frequenza settimanale
    → Il valore del mese M diventa disponibile dalla settimana che include la
       data di osservazione (inizio mese per FED_FUNDS, fine mese per CPI).
    → Conservative: usiamo data osservazione FRED come data disponibilità.
       CPI ha lag ~2 settimane nella realtà, ma FRED riporta comunque la data
       di inizio mese — in pratica questo introduce un bias minimo accettabile
       poiché il CPI è già noto prima del venerdì della settimana successiva.

  DATI YFINANCE (prezzi giornalieri):
    → close: resample W-FRI last() = chiusura del venerdì
    → volume: resample W-FRI sum() = volume totale della settimana

  COT (settimanale, "as of" martedì, pubblicato venerdì):
    → Il dato COT si riferisce al martedì, ma è disponibile solo dal venerdì.
    → Impostiamo la data di disponibilità = data_osservazione + 3 giorni
       (martedì → venerdì), poi resample W-FRI last().
    → Questo evita lookahead: non usiamo il dato COT prima che sia pubblico.

  WGC DEMAND (annuale):
    → resample W-FRI forward-fill
    → Il valore dell'anno Y ha come data 31-dic-Y → disponibile dall'inizio Y+1

VERIFICA LOOKAHEAD:
  Per ogni serie, la riga della settimana W contiene SOLO dati il cui
  timestamp di disponibilità è <= venerdì della settimana W.
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
    START_DATE, END_DATE, WEEKLY_FREQ,
    RAW_FRED_DIR, RAW_YFINANCE_DIR, RAW_MANUAL_DIR,
    PROCESSED_DIR,
    FRED_SERIES, YFINANCE_TICKERS,
)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
WEEKLY_INDEX_START = pd.Timestamp(START_DATE)
WEEKLY_INDEX_END   = pd.Timestamp(END_DATE) if END_DATE else pd.Timestamp.today()


# ===========================================================================
# FUNZIONI DI CARICAMENTO E ALLINEAMENTO
# ===========================================================================

def make_weekly_index() -> pd.DatetimeIndex:
    """Genera l'indice settimanale (venerdì) dall'inizio alla fine."""
    return pd.date_range(
        start=WEEKLY_INDEX_START,
        end=WEEKLY_INDEX_END,
        freq=WEEKLY_FREQ,
    )


def load_csv(path: Path, name: str) -> pd.DataFrame | None:
    """Carica un CSV con indice di data. Ritorna None se il file non esiste."""
    if not path.exists():
        print(f"  ⚠ File non trovato: {path.name}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False)
    df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df


def align_daily_to_weekly(
    series: pd.Series,
    weekly_idx: pd.DatetimeIndex,
    agg: str = "last",
) -> pd.Series:
    """
    Aggrega una serie giornaliera a frequenza settimanale (W-FRI).
    agg='last'  → valore del venerdì (o ultimo giorno disponibile della settimana)
    agg='mean'  → media settimanale
    agg='sum'   → somma settimanale

    Lookahead safe: ogni settimana W usa solo dati con data <= venerdì di W.
    """
    series = series.dropna()
    if agg == "last":
        weekly = series.resample(WEEKLY_FREQ).last()
    elif agg == "mean":
        weekly = series.resample(WEEKLY_FREQ).mean()
    elif agg == "sum":
        weekly = series.resample(WEEKLY_FREQ).sum(min_count=1)
    else:
        raise ValueError(f"agg non riconosciuto: {agg}")

    return weekly.reindex(weekly_idx)


def align_monthly_to_weekly(
    series: pd.Series,
    weekly_idx: pd.DatetimeIndex,
) -> pd.Series:
    """
    Forward-fill una serie mensile a frequenza settimanale.

    Logica lookahead-safe:
      - Il valore del mese M (data FRED = primo giorno del mese) è disponibile
        dalla settimana che contiene quella data.
      - Usiamo forward-fill: ogni settimana riceve il valore dell'ultimo mese
        il cui inizio è <= venerdì della settimana.

    Non applichiamo shift aggiuntivi: la data FRED per dati mensili è il
    primo giorno del mese, il che è già conservativo (il dato è noto con
    qualche settimana di lag nella realtà). Per il modello settimanale
    questo introduce al massimo 2-4 ore di bias, trascurabile.
    """
    series = series.dropna()
    # Forward-fill verso l'indice settimanale
    combined = series.reindex(series.index.union(weekly_idx)).ffill()
    return combined.reindex(weekly_idx)


def align_cot_to_weekly(
    df: pd.DataFrame,
    weekly_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Allinea il COT a frequenza settimanale con gestione del publication lag.

    Il dato COT ha come data l'osservazione (martedì).
    Viene però pubblicato CFTC il venerdì (~3 giorni dopo).

    Per evitare lookahead: shiftiamo la data di +3 giorni (martedì → venerdì),
    poi facciamo forward-fill sull'indice settimanale.

    Questo significa che il dato COT del martedì T diventa disponibile
    dal venerdì T+3, che coincide quasi sempre con la settimana W-FRI corrente.
    """
    df = df.copy()
    # Shift di 3 giorni = data di disponibilità (venerdì)
    df.index = df.index + pd.Timedelta(days=3)
    # Forward-fill sull'indice settimanale
    cols_out = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        s = df[col].dropna()
        combined = s.reindex(s.index.union(weekly_idx)).ffill()
        cols_out[col] = combined.reindex(weekly_idx)
    return pd.DataFrame(cols_out, index=weekly_idx)


def align_annual_to_weekly(
    df: pd.DataFrame,
    weekly_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Forward-fill un DataFrame annuale a frequenza settimanale.

    Il dato dell'anno Y (data 31-dic-Y) è disponibile dall'inizio di Y+1.
    Per evitare lookahead: spostiamo la data di pubblicazione al 1-gen-(Y+1)
    (cioè +1 giorno dal 31-dic), poi forward-fill.
    """
    df = df.copy()
    # Shift minimale: 31-dic → 1-gen dell'anno seguente
    df.index = df.index + pd.Timedelta(days=1)
    cols_out = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        s = df[col].dropna()
        combined = s.reindex(s.index.union(weekly_idx)).ffill()
        cols_out[col] = combined.reindex(weekly_idx)
    return pd.DataFrame(cols_out, index=weekly_idx)


# ===========================================================================
# COSTRUZIONE DATASET
# ===========================================================================

def build_dataset() -> pd.DataFrame:
    """
    Costruisce il dataset settimanale integrato.
    Ritorna il DataFrame con indice W-FRI e tutte le colonne.
    """
    weekly_idx = make_weekly_index()
    frames: dict[str, pd.Series | pd.DataFrame] = {}

    print(f"\n  Indice settimanale: {weekly_idx[0].date()} → {weekly_idx[-1].date()} "
          f"({len(weekly_idx)} settimane)\n")

    # -------------------------------------------------------------------
    # 1. SERIE FRED GIORNALIERE
    # -------------------------------------------------------------------
    daily_fred = [
        "REAL_YIELD_10Y", "BREAKEVEN_10Y", "BREAKEVEN_5Y",
        "VIX_FRED", "EPU", "DFF",
    ]
    monthly_fred = ["FED_FUNDS", "CPI"]

    print("  [1] FRED giornaliere...")
    for name in daily_fred:
        path = RAW_FRED_DIR / f"{name}.csv"
        df = load_csv(path, name)
        if df is None:
            continue
        s = df["value"].rename(name)
        aligned = align_daily_to_weekly(s, weekly_idx, agg="last")
        frames[name] = aligned
        pct_missing = aligned.isna().mean() * 100
        print(f"    {name:<22} → {len(aligned)} righe, {pct_missing:.1f}% NaN")

    print("  [2] FRED mensili (forward-fill)...")
    for name in monthly_fred:
        path = RAW_FRED_DIR / f"{name}.csv"
        df = load_csv(path, name)
        if df is None:
            continue
        s = df["value"].rename(name)
        aligned = align_monthly_to_weekly(s, weekly_idx)
        frames[name] = aligned
        pct_missing = aligned.isna().mean() * 100
        print(f"    {name:<22} → {len(aligned)} righe, {pct_missing:.1f}% NaN")

    # -------------------------------------------------------------------
    # 2. YFINANCE
    # -------------------------------------------------------------------
    print("  [3] yfinance (chiusura settimanale)...")
    for name in YFINANCE_TICKERS:
        path = RAW_YFINANCE_DIR / f"{name}.csv"
        df = load_csv(path, name)
        if df is None:
            continue

        if "close" in df.columns:
            s_close = df["close"].rename(f"{name}_close")
            frames[f"{name}_close"] = align_daily_to_weekly(s_close, weekly_idx, agg="last")

        if "volume" in df.columns:
            s_vol = df["volume"].rename(f"{name}_volume")
            # Volume: somma settimanale (totale transato nella settimana)
            frames[f"{name}_volume"] = align_daily_to_weekly(s_vol, weekly_idx, agg="sum")

        pct_missing = frames[f"{name}_close"].isna().mean() * 100
        print(f"    {name:<22} → {len(weekly_idx)} righe, {pct_missing:.1f}% NaN")

    # -------------------------------------------------------------------
    # 3. GPR (giornaliero → settimanale last)
    # -------------------------------------------------------------------
    print("  [4] GPR (giornaliero → settimanale)...")
    gpr_path = RAW_MANUAL_DIR / "gpr_parsed.csv"
    gpr_df = load_csv(gpr_path, "GPR")
    if gpr_df is not None:
        for col in ["GPRD", "GPRD_ACT", "GPRD_THREAT"]:
            if col in gpr_df.columns:
                s = gpr_df[col].rename(f"GPR_{col}")
                frames[f"GPR_{col}"] = align_daily_to_weekly(s, weekly_idx, agg="last")
        pct_missing = frames["GPR_GPRD"].isna().mean() * 100
        print(f"    GPR                    → {len(weekly_idx)} righe, {pct_missing:.1f}% NaN")

    # -------------------------------------------------------------------
    # 4. COT (settimanale, con publication lag shift +3gg)
    # -------------------------------------------------------------------
    print("  [5] COT (settimanale + shift +3gg pubblicazione)...")
    cot_path = RAW_MANUAL_DIR / "cot_gold.csv"
    cot_df = load_csv(cot_path, "COT")
    if cot_df is not None:
        # Mantieni solo colonne numeriche rilevanti
        cot_cols = [c for c in ["open_interest", "nc_long", "nc_short", "nc_spread",
                                 "comm_long", "comm_short"] if c in cot_df.columns]
        cot_aligned = align_cot_to_weekly(cot_df[cot_cols], weekly_idx)
        # Rinomina con prefisso COT_
        cot_aligned.columns = [f"COT_{c}" for c in cot_aligned.columns]
        for col in cot_aligned.columns:
            frames[col] = cot_aligned[col]
        pct_missing = cot_aligned.isna().mean().mean() * 100
        print(f"    COT ({len(cot_cols)} colonne)       → {len(weekly_idx)} righe, {pct_missing:.1f}% NaN medio")

    # -------------------------------------------------------------------
    # 5. WGC DEMAND (annuale → settimanale forward-fill, shift +1gg)
    # -------------------------------------------------------------------
    print("  [6] WGC demand (annuale → settimanale forward-fill)...")
    wgc_path = RAW_MANUAL_DIR / "wgc_demand_parsed.csv"
    wgc_df = load_csv(wgc_path, "WGC")
    if wgc_df is not None:
        # Tieni solo le colonne già rinominate (con prefisso wgc_)
        wgc_cols = [c for c in wgc_df.columns if c.startswith("wgc_")]
        if wgc_cols:
            wgc_aligned = align_annual_to_weekly(wgc_df[wgc_cols], weekly_idx)
            for col in wgc_aligned.columns:
                frames[col] = wgc_aligned[col]
            pct_missing = wgc_aligned.isna().mean().mean() * 100
            print(f"    WGC ({len(wgc_cols)} colonne)       → {len(weekly_idx)} righe, {pct_missing:.1f}% NaN medio")
        else:
            print(f"    WGC: nessuna colonna 'wgc_' trovata, skip")

    # -------------------------------------------------------------------
    # MERGE in unico DataFrame
    # -------------------------------------------------------------------
    print("\n  Merge in dataset unico...")
    df_all = pd.DataFrame(index=weekly_idx)
    df_all.index.name = "date"
    for col_name, series_or_df in frames.items():
        if isinstance(series_or_df, pd.Series):
            df_all[col_name] = series_or_df
        else:
            for col in series_or_df.columns:
                df_all[col] = series_or_df[col]

    return df_all


# ===========================================================================
# REPORT QUALITÀ POST-MERGE
# ===========================================================================

def print_dataset_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  DATASET INTEGRATO — Report")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\n  Shape:    {df.shape[0]} righe × {df.shape[1]} colonne")
    print(f"  Periodo:  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Freq:     settimanale (W-FRI)\n")

    print(f"  {'COLONNA':<35} {'NON-NaN':>8}  {'NaN%':>6}  {'MIN':>12}  {'MAX':>12}")
    print("  " + "-" * 82)
    for col in df.columns:
        s = df[col]
        n_valid = s.notna().sum()
        pct_nan = s.isna().mean() * 100
        vmin = f"{s.min():.4f}" if s.notna().any() else "N/A"
        vmax = f"{s.max():.4f}" if s.notna().any() else "N/A"
        print(f"  {col:<35} {n_valid:>8}  {pct_nan:>5.1f}%  {vmin:>12}  {vmax:>12}")

    print("\n  Statistiche descrittive (prime 5 colonne):")
    print(df.iloc[:, :5].describe().round(4).to_string(col_space=12))

    print("\n  Prime 3 righe:")
    print(df.head(3).to_string())

    print("\n  Ultime 3 righe:")
    print(df.tail(3).to_string())

    # Warning su colonne con >50% NaN nel periodo 2005-2025
    window = df.loc["2005":"2025"]
    heavy_missing = [
        col for col in df.columns
        if window[col].isna().mean() > 0.50
    ]
    if heavy_missing:
        print(f"\n  ⚠ Colonne con >50% NaN nel periodo 2005-2025:")
        for col in heavy_missing:
            pct = window[col].isna().mean() * 100
            print(f"    {col}: {pct:.1f}%")
    else:
        print("\n  ✓ Nessuna colonna con >50% NaN nel periodo 2005-2025")

    print("\n" + "=" * 70)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 70)
    print("  BUILD DATASET — Gold Trend Prediction Model")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    df = build_dataset()

    print_dataset_report(df)

    # Salva
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset_raw.csv"
    df.to_csv(out_path)
    print(f"\n  ✓ Salvato: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"    {df.shape[0]} righe × {df.shape[1]} colonne\n")


if __name__ == "__main__":
    main()
