"""
feature_engineering.py — Costruzione feature predittive dal dataset raw settimanale

Esecuzione:
    cd gold_model
    python -m src.features.feature_engineering

Output:
    /data/processed/dataset_features.csv

Struttura delle feature per ogni variabile base X:
  - X                          → valore grezzo (livello)
  - X_chg_4w  / X_pct_4w      → variazione assoluta / percentuale a 4 settimane
  - X_chg_8w  / X_pct_8w      → idem a 8 settimane
  - X_chg_12w / X_pct_12w     → idem a 12 settimane
  - X_chg_26w / X_pct_26w     → idem a 26 settimane
  - X_vs_ma52                  → distanza % dalla media mobile a 52 settimane
  - X_above_ma52               → 1 se sopra la MA52, 0 altrimenti
  - X_pctile_3y                → percentile rolling su 3 anni (156 settimane)

Feature specifiche:
  COT:
    - COT_net_position          → nc_long - nc_short (nette speculatori)
    - COT_net_pct_oi            → net / open_interest (normalizzato)
    - COT_net_pctile_3y         → percentile rolling net position (contrarian)

  VIX:
    - VIX_above_20 / VIX_above_30 → regime binario
    - VIX_regime                   → 0=basso(<20), 1=medio(20-30), 2=alto(>30)

  TIPS yield (REAL_YIELD_10Y):
    - REAL_YIELD_falling_4w     → 1 se in calo da almeno 4 settimane consecutive
    - REAL_YIELD_falling_8w     → 1 se in calo da almeno 8 settimane consecutive

  DXY:
    - DXY_mom_4w / DXY_mom_12w  → momentum a 4 e 12 settimane (già in pct_Xw)

  ETF Gold (GLD):
    - GLD_volume_ma4            → media mobile volume a 4 settimane
    - GLD_volume_vs_ma26        → volume corrente / MA26 (surge detector)

TARGET (costruiti guardando avanti — usare solo DOPO il training split):
  - gold_fwd_12w   → variazione % prezzo oro a +12 settimane (continua)
  - gold_fwd_16w   → variazione % prezzo oro a +16 settimane (continua)
  - gold_fwd_26w   → variazione % prezzo oro a +26 settimane (continua)
  - target_12w     → 1 se gold_fwd_12w >= +2%, 0 altrimenti
  - target_16w     → 1 se gold_fwd_16w >= +2%, 0 altrimenti (TARGET PRINCIPALE)
  - target_26w     → 1 se gold_fwd_26w >= +2%, 0 altrimenti
  - target_16w_3cls → 3 classi: 2=long(>+2%), 1=flat(-2%..+2%), 0=short(<-2%)

ATTENZIONE LOOKAHEAD:
  I target guardano AVANTI (shift negativo) — devono essere usati solo per
  la riga della settimana W che predice W+N.
  Nel training, assicurarsi che per ogni settimana W, i target di W vengano
  usati solo come Y, mai come X. Il codice li separa esplicitamente.
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
    PROCESSED_DIR,
    TARGET_HORIZONS_WEEKS,
    TARGET_THRESHOLD_PCT,
    ROLLING_CORR_WINDOW,
)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
LAGS_WEEKS     = [4, 8, 12, 16, 26]
MA_LONG_WINDOW = 52   # Media mobile lunga (1 anno)
PCTILE_WINDOW  = 156  # Finestra percentile rolling (3 anni)


# ===========================================================================
# TRASFORMAZIONI DI BASE
# ===========================================================================

def add_level_trend_features(df_out: pd.DataFrame, series: pd.Series, name: str) -> None:
    """
    Aggiunge al DataFrame df_out le feature di livello e trend per una serie.
    Modifica df_out in-place.

    Feature aggiunte:
      - {name}                  (valore grezzo, già presente se passato dal raw)
      - {name}_chg_{N}w         variazione assoluta a N settimane
      - {name}_pct_{N}w         variazione percentuale a N settimane
      - {name}_vs_ma52          distanza % dalla MA52
      - {name}_above_ma52       1 se sopra la MA52
      - {name}_pctile_3y        percentile rolling 3 anni
    """
    # Valore grezzo
    df_out[name] = series

    # Variazioni assolute e percentuali per ogni lag
    for lag in LAGS_WEEKS:
        df_out[f"{name}_chg_{lag}w"]  = series - series.shift(lag)
        df_out[f"{name}_pct_{lag}w"]  = series.pct_change(lag) * 100

    # Media mobile 52 settimane e posizione relativa
    ma52 = series.rolling(MA_LONG_WINDOW, min_periods=26).mean()
    df_out[f"{name}_vs_ma52"]    = (series - ma52) / ma52.abs().replace(0, np.nan) * 100
    df_out[f"{name}_above_ma52"] = (series > ma52).astype(float)

    # Percentile rolling su 3 anni (0-100)
    def rolling_percentile(x: pd.Series, window: int) -> pd.Series:
        """Calcola il percentile del valore corrente nella finestra rolling."""
        return x.rolling(window, min_periods=window // 2).apply(
            lambda w: float(pd.Series(w).rank(pct=True).iloc[-1] * 100),
            raw=False,
        )

    df_out[f"{name}_pctile_3y"] = rolling_percentile(series, PCTILE_WINDOW)


# ===========================================================================
# FEATURE SPECIFICHE PER VARIABILE
# ===========================================================================

def add_cot_features(df_out: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Feature specifiche per il COT."""
    if "COT_nc_long" not in df_raw.columns or "COT_nc_short" not in df_raw.columns:
        return

    nc_long  = df_raw["COT_nc_long"]
    nc_short = df_raw["COT_nc_short"]
    oi       = df_raw.get("COT_open_interest", pd.Series(dtype=float))

    # Net position speculatori (long - short)
    net = nc_long - nc_short
    df_out["COT_net_position"]   = net
    df_out["COT_net_chg_4w"]     = net - net.shift(4)
    df_out["COT_net_chg_8w"]     = net - net.shift(8)

    # Net position normalizzata per open interest
    if oi.notna().any():
        df_out["COT_net_pct_oi"] = net / oi.replace(0, np.nan) * 100

    # Percentile rolling 3 anni della net position (segnale contrarian)
    # Quando è a valori estremi (>90 o <10), il mercato è troppo posizionato
    def rolling_pctile(x: pd.Series) -> pd.Series:
        return x.rolling(PCTILE_WINDOW, min_periods=PCTILE_WINDOW // 2).apply(
            lambda w: float(pd.Series(w).rank(pct=True).iloc[-1] * 100),
            raw=False,
        )

    df_out["COT_net_pctile_3y"]  = rolling_pctile(net)

    # Ratio long/short (altro indicatore di estremo di posizionamento)
    df_out["COT_long_short_ratio"] = nc_long / nc_short.replace(0, np.nan)


def add_vix_features(df_out: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Feature specifiche per il VIX."""
    # Usa VIX_FRED come fonte principale (più completa)
    vix_col = "VIX_FRED" if "VIX_FRED" in df_raw.columns else "VIX_YF_close"
    if vix_col not in df_raw.columns:
        return

    vix = df_raw[vix_col]

    # Regime binario
    df_out["VIX_above_20"] = (vix > 20).astype(float)
    df_out["VIX_above_30"] = (vix > 30).astype(float)
    df_out["VIX_regime"]   = pd.cut(
        vix, bins=[-np.inf, 20, 30, np.inf], labels=[0, 1, 2]
    ).astype(float)

    # Variazione VIX a 4 settimane (spike detector)
    df_out["VIX_chg_4w"]  = vix - vix.shift(4)


def add_tips_features(df_out: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Feature specifiche per TIPS real yield (relazione inversa con oro)."""
    if "REAL_YIELD_10Y" not in df_raw.columns:
        return

    tips = df_raw["REAL_YIELD_10Y"]

    # Direzione del trend: quante settimane consecutive in calo/rialzo
    direction = np.sign(tips.diff())   # +1=salita, -1=discesa, 0=invariato

    def count_consecutive(dir_series: pd.Series, sign: float) -> pd.Series:
        """Conta quante settimane consecutive il segno è uguale a 'sign'."""
        match = (dir_series == sign).astype(int)
        counts = []
        count = 0
        for v in match:
            if v == 1:
                count += 1
            else:
                count = 0
            counts.append(count)
        return pd.Series(counts, index=dir_series.index)

    falling_weeks = count_consecutive(direction, -1.0)

    df_out["REAL_YIELD_falling_4w"] = (falling_weeks >= 4).astype(float)
    df_out["REAL_YIELD_falling_8w"] = (falling_weeks >= 8).astype(float)
    df_out["REAL_YIELD_falling_weeks"] = falling_weeks


def add_etf_volume_features(df_out: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Feature specifiche per volume GLD (proxy flussi ETF)."""
    if "GLD_volume" not in df_raw.columns:
        return

    vol = df_raw["GLD_volume"].replace(0, np.nan)

    ma4  = vol.rolling(4,  min_periods=2).mean()
    ma26 = vol.rolling(26, min_periods=13).mean()

    df_out["GLD_volume_ma4"]      = ma4
    df_out["GLD_volume_vs_ma26"]  = vol / ma26.replace(0, np.nan)  # ratio > 1 = surge


def add_inflation_spread_features(df_out: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Feature composte sull'inflazione e spread tra serie."""
    # Spread breakeven 10Y - 5Y (inclinazione della curva breakeven)
    if "BREAKEVEN_10Y" in df_raw.columns and "BREAKEVEN_5Y" in df_raw.columns:
        df_out["BREAKEVEN_spread"] = df_raw["BREAKEVEN_10Y"] - df_raw["BREAKEVEN_5Y"]

    # Real rate vs CPI: rapporto tra tasso reale e inflazione attesa
    if "REAL_YIELD_10Y" in df_raw.columns and "BREAKEVEN_10Y" in df_raw.columns:
        # Tasso nominale implicito = real yield + breakeven
        df_out["NOMINAL_YIELD_implied"] = df_raw["REAL_YIELD_10Y"] + df_raw["BREAKEVEN_10Y"]

    # CPI variazione annua (proxy inflazione realizzata YoY)
    if "CPI" in df_raw.columns:
        cpi = df_raw["CPI"]
        df_out["CPI_yoy_pct"] = cpi.pct_change(52) * 100   # ~52 settimane = 1 anno


# ===========================================================================
# TARGET
# ===========================================================================

def add_targets(df_out: pd.DataFrame, gold_price: pd.Series) -> None:
    """
    Aggiunge le colonne target al DataFrame.

    ATTENZIONE: queste colonne guardano AVANTI nel tempo.
    Devono essere usate SOLO come variabile dipendente (Y) nel training,
    mai come feature (X).

    I target per la settimana W sono calcolati come:
        gold_fwd_Nw[W] = (gold_price[W+N] / gold_price[W] - 1) * 100
    """
    for horizon in TARGET_HORIZONS_WEEKS:
        # Variazione percentuale continua (float)
        fwd_pct = gold_price.shift(-horizon) / gold_price - 1
        fwd_pct *= 100
        df_out[f"gold_fwd_{horizon}w"] = fwd_pct

        # Target binario: ≥ soglia → 1 (rialzista), altrimenti 0
        df_out[f"target_{horizon}w"] = (fwd_pct >= TARGET_THRESHOLD_PCT * 100).astype(float)
        # Dove non c'è abbastanza storico futuro, lascia NaN
        df_out.loc[fwd_pct.isna(), f"target_{horizon}w"] = np.nan

    # Target a 3 classi per il target principale (16 settimane)
    threshold = TARGET_THRESHOLD_PCT * 100
    fwd_16w = df_out["gold_fwd_16w"]
    conditions = [
        fwd_16w >= threshold,          # long
        fwd_16w <= -threshold,         # short
    ]
    choices = [2, 0]
    df_out["target_16w_3cls"] = np.select(conditions, choices, default=1)  # default=flat
    df_out.loc[fwd_16w.isna(), "target_16w_3cls"] = np.nan


# ===========================================================================
# FUNZIONE PRINCIPALE
# ===========================================================================

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce tutte le feature dal DataFrame raw.
    Ritorna un nuovo DataFrame con feature e target.
    """
    df_out = pd.DataFrame(index=df_raw.index)

    # -----------------------------------------------------------------------
    # 1. Feature di livello e trend per ogni variabile macro principale
    # -----------------------------------------------------------------------
    macro_vars = [
        # (colonna_raw, nome_feature)
        ("REAL_YIELD_10Y",    "REAL_YIELD_10Y"),
        ("BREAKEVEN_10Y",     "BREAKEVEN_10Y"),
        ("BREAKEVEN_5Y",      "BREAKEVEN_5Y"),
        ("VIX_FRED",          "VIX"),
        ("EPU",               "EPU"),
        ("DFF",               "DFF"),
        ("FED_FUNDS",         "FED_FUNDS"),
        ("CPI",               "CPI"),
        ("DXY_close",         "DXY"),
        ("TLT_close",         "TLT"),
        ("TNX_close",         "TNX"),
        ("GPR_GPRD",          "GPR"),
        ("GPR_GPRD_ACT",      "GPR_ACT"),
        ("GPR_GPRD_THREAT",   "GPR_THREAT"),
        ("GLD_close",         "GLD"),
        ("IAU_close",         "IAU"),
        ("GOLD_FUTURES_close","GOLD"),
        ("COT_open_interest", "COT_OI"),
        ("COT_nc_long",       "COT_NC_LONG"),
        ("COT_nc_short",      "COT_NC_SHORT"),
        # WGC annuali — solo livello + 1 lag (dati troppo lenti per lag multipli)
        ("wgc_total_demand",         "WGC_DEMAND"),
        ("wgc_investment",           "WGC_INVEST"),
        ("wgc_etf_flows",            "WGC_ETF"),
        ("wgc_central_bank_demand",  "WGC_CB"),
    ]

    print("  Costruzione feature livello/trend/percentile...")
    for raw_col, feat_name in macro_vars:
        if raw_col not in df_raw.columns:
            continue
        series = df_raw[raw_col]
        add_level_trend_features(df_out, series, feat_name)
        print(f"    ✓ {feat_name:<25} ({len([c for c in df_out.columns if c.startswith(feat_name)])} feature)")

    # -----------------------------------------------------------------------
    # 2. Feature specifiche per variabile
    # -----------------------------------------------------------------------
    print("  Costruzione feature specifiche...")
    add_cot_features(df_out, df_raw)
    add_vix_features(df_out, df_raw)
    add_tips_features(df_out, df_raw)
    add_etf_volume_features(df_out, df_raw)
    add_inflation_spread_features(df_out, df_raw)
    print(f"    ✓ COT net, VIX regime, TIPS trend, ETF volume, spread inflazione")

    # -----------------------------------------------------------------------
    # 3. Feature di FSI (già settimanale, solo level+trend)
    # -----------------------------------------------------------------------
    if "FSI" in df_raw.columns:
        add_level_trend_features(df_out, df_raw["FSI"], "FSI")
        print(f"    ✓ FSI")

    # -----------------------------------------------------------------------
    # 3b. Pulizia valori infiniti: alcune serie (real yield, breakeven)
    #     attraversano lo zero → pct_change produce ±inf → NaN è più corretto
    # -----------------------------------------------------------------------
    feat_cols = [c for c in df_out.columns
                 if not c.startswith("target_") and not c.startswith("gold_fwd_")]
    inf_count = np.isinf(df_out[feat_cols].values).sum()
    if inf_count > 0:
        df_out[feat_cols] = df_out[feat_cols].replace([np.inf, -np.inf], np.nan)
        print(f"    ⚠  {inf_count} valori ±inf sostituiti con NaN (serie che attraversano lo zero)")

    # -----------------------------------------------------------------------
    # 4. Target (N.B.: guardano avanti — non usare come X)
    # -----------------------------------------------------------------------
    print("  Costruzione target...")
    gold_price = df_raw["GOLD_FUTURES_close"]
    add_targets(df_out, gold_price)
    target_cols = [c for c in df_out.columns if c.startswith("target_") or c.startswith("gold_fwd_")]
    print(f"    ✓ {len(target_cols)} colonne target: {target_cols}")

    return df_out


# ===========================================================================
# ANALISI DI CORRELAZIONE CON IL TARGET 16W
# ===========================================================================

def compute_feature_target_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la correlazione di Pearson tra ogni feature e target_16w.
    Esclude le colonne target e gold_fwd dalle feature.
    Ritorna DataFrame ordinato per |correlazione| decrescente.
    """
    target = df["target_16w"].dropna()
    exclude = {c for c in df.columns if c.startswith("target_") or c.startswith("gold_fwd_")}
    feature_cols = [c for c in df.columns if c not in exclude]

    rows = []
    for col in feature_cols:
        x = df[col]
        # Allinea sugli indici comuni (dove entrambi non sono NaN)
        common = x.dropna().index.intersection(target.index)
        if len(common) < 50:
            continue
        corr = x.loc[common].corr(target.loc[common])
        rows.append({"feature": col, "pearson_r": corr, "abs_r": abs(corr), "n_obs": len(common)})

    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


# ===========================================================================
# REPORT
# ===========================================================================

def print_report(df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    target_cols = [c for c in df.columns if c.startswith("target_") or c.startswith("gold_fwd_")]
    feature_cols = [c for c in df.columns if c not in set(target_cols)]

    n_total   = len(df)
    n_train   = (df.index < "2016-01-01").sum()
    # Righe utili per il target principale (16w): escludi ultime 16 settimane
    n_target = df["target_16w"].notna().sum()

    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING — Report")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\n  Righe totali:                {n_total}")
    print(f"  Righe con target_16w valido: {n_target} (ultime 16 settimane escluse)")
    print(f"  Feature costruite:           {len(feature_cols)}")
    print(f"  Colonne target:              {len(target_cols)}")
    print(f"\n  Target_16w distribuzione (periodo 2005-2025):")
    t16 = df.loc["2005":"2025", "target_16w"].dropna()
    vc = t16.value_counts().sort_index()
    for k, v in vc.items():
        label = "Rialzista (≥+2%)" if k == 1.0 else "Non rialzista"
        print(f"    {label}: {v} ({v/len(t16)*100:.1f}%)")

    print(f"\n  Target_16w_3cls distribuzione (2005-2025):")
    t3 = df.loc["2005":"2025", "target_16w_3cls"].dropna()
    vc3 = t3.value_counts().sort_index()
    labels3 = {0.0: "Short (<-2%)", 1.0: "Flat", 2.0: "Long (≥+2%)"}
    for k, v in vc3.items():
        print(f"    {labels3.get(k, k)}: {v} ({v/len(t3)*100:.1f}%)")

    print(f"\n  Top 30 feature per correlazione con target_16w (Pearson |r|):")
    print(f"  {'#':<4} {'FEATURE':<40} {'r':>7}  {'N':>6}")
    print("  " + "-" * 62)
    for i, row in corr_df.head(30).iterrows():
        direction = "▲" if row["pearson_r"] > 0 else "▼"
        print(f"  {i+1:<4} {row['feature']:<40} {row['pearson_r']:>+7.4f} {direction}  {row['n_obs']:>6}")

    # Colonne con >50% NaN
    heavy_missing = [(c, df[c].isna().mean()*100) for c in feature_cols if df[c].isna().mean() > 0.30]
    if heavy_missing:
        print(f"\n  Colonne con >30% NaN:")
        for col, pct in sorted(heavy_missing, key=lambda x: -x[1]):
            print(f"    {col}: {pct:.1f}%")

    print("\n" + "=" * 70)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 70)
    print("  FEATURE ENGINEERING — Gold Trend Prediction Model")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Carica dataset raw
    raw_path = PROCESSED_DIR / "dataset_raw.csv"
    if not raw_path.exists():
        print(f"\n  ERRORE: file non trovato: {raw_path}")
        print("  Eseguire prima: python -m src.data.build_dataset")
        sys.exit(1)

    print(f"\n  Caricamento dataset raw: {raw_path.name}")
    df_raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df_raw.index = pd.to_datetime(df_raw.index, utc=False)
    df_raw.index = df_raw.index.tz_localize(None)
    print(f"  Shape: {df_raw.shape}\n")

    # Costruzione feature
    df_feat = build_features(df_raw)

    # Correlazione con target
    print("\n  Calcolo correlazione feature-target (Pearson)...")
    corr_df = compute_feature_target_correlation(df_feat)

    # Report
    print_report(df_feat, corr_df)

    # Salvataggio
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset_features.csv"
    df_feat.to_csv(out_path)

    corr_path = PROCESSED_DIR / "feature_target_correlation.csv"
    corr_df.to_csv(corr_path, index=False)

    n_features = len([c for c in df_feat.columns
                      if not c.startswith("target_") and not c.startswith("gold_fwd_")])
    print(f"\n  ✓ Salvato: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"    {df_feat.shape[0]} righe × {df_feat.shape[1]} colonne totali ({n_features} feature + target)")
    print(f"  ✓ Salvato: {corr_path.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
