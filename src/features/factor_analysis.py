"""
factor_analysis.py — Analisi fattoriale e selezione variabili

Esecuzione:
    cd gold_model
    python -m src.features.factor_analysis

Obiettivo:
    Ridurre le 305 feature → 40-60 feature candidate per il modello ensemble,
    rimuovendo ridondanza inter-feature mantenendo la capacità predittiva.

REGOLA ANTI-LOOKAHEAD:
    Tutta la selezione avviene SOLO sul set di training (2005-01-01 → 2015-12-31).
    Il processo di selezione non vede mai i dati di test (2016+).

Pipeline di selezione:
    1. Definizione gruppi tematici (8 famiglie)
    2. Filtro varianza bassa (std < soglia sul train set)
    3. All'interno di ogni gruppo: pruning per correlazione intra-gruppo
       (|r| > 0.85 → tieni quello con |r_target| più alta)
    4. Filtro univariato globale: tieni le N_TOP feature per gruppo
    5. Pruning globale cross-gruppo: rimuovi feature con |r| > 0.80
       già coperte da un'altra più predittiva
    6. Aggiunta feature speciali sempre incluse (VIX regime, TIPS trend,
       COT net percentile, DXY momentum)

Output:
    data/processed/selected_features.txt    — lista feature selezionate
    data/processed/dataset_selected.csv     — dataset finale (feature + target)
    data/processed/factor_analysis_report.csv — dettaglio correlazioni
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
from itertools import combinations

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

from config import PROCESSED_DIR

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
TRAIN_START = "2005-01-01"
TRAIN_END   = "2015-12-31"

INTRA_GROUP_CORR_THRESHOLD  = 0.85  # Pruning intra-gruppo
CROSS_GROUP_CORR_THRESHOLD  = 0.80  # Pruning cross-gruppo
MIN_OBS_FOR_CORR            = 100   # Minimo osservazioni per calcolare correlazione valida
MIN_STD                     = 0.001 # Filtro varianza bassa
N_TOP_PER_GROUP             = 8     # Feature top da mantenere per ogni gruppo


# ===========================================================================
# GRUPPI TEMATICI
# ===========================================================================

def define_feature_groups() -> dict[str, list[str]]:
    """
    Mappa ogni feature al suo gruppo tematico.
    Feature non in nessun gruppo → gruppo 'other'.
    """
    groups = {

        # ----- 1. Tassi reali e inflazione -----
        "rates_real": [
            "REAL_YIELD_10Y",
            "REAL_YIELD_10Y_chg_4w",  "REAL_YIELD_10Y_chg_8w",
            "REAL_YIELD_10Y_chg_12w", "REAL_YIELD_10Y_chg_16w", "REAL_YIELD_10Y_chg_26w",
            "REAL_YIELD_10Y_pct_4w",  "REAL_YIELD_10Y_pct_8w",
            "REAL_YIELD_10Y_pct_12w", "REAL_YIELD_10Y_pct_16w", "REAL_YIELD_10Y_pct_26w",
            "REAL_YIELD_10Y_vs_ma52", "REAL_YIELD_10Y_above_ma52",
            "REAL_YIELD_10Y_pctile_3y",
            "REAL_YIELD_falling_4w",  "REAL_YIELD_falling_8w",
            "REAL_YIELD_falling_weeks",
            "NOMINAL_YIELD_implied",
        ],

        # ----- 2. Breakeven / inflazione attesa -----
        "rates_inflation": [
            "BREAKEVEN_10Y",
            "BREAKEVEN_10Y_chg_4w",  "BREAKEVEN_10Y_chg_8w",
            "BREAKEVEN_10Y_chg_12w", "BREAKEVEN_10Y_chg_16w", "BREAKEVEN_10Y_chg_26w",
            "BREAKEVEN_10Y_pct_4w",  "BREAKEVEN_10Y_pct_8w",
            "BREAKEVEN_10Y_pct_12w", "BREAKEVEN_10Y_pct_16w", "BREAKEVEN_10Y_pct_26w",
            "BREAKEVEN_10Y_vs_ma52", "BREAKEVEN_10Y_above_ma52",
            "BREAKEVEN_10Y_pctile_3y",
            "BREAKEVEN_5Y",
            "BREAKEVEN_5Y_chg_4w",   "BREAKEVEN_5Y_chg_8w",
            "BREAKEVEN_5Y_chg_12w",  "BREAKEVEN_5Y_chg_16w",  "BREAKEVEN_5Y_chg_26w",
            "BREAKEVEN_5Y_pct_4w",   "BREAKEVEN_5Y_pct_8w",
            "BREAKEVEN_5Y_pct_12w",  "BREAKEVEN_5Y_pct_16w",  "BREAKEVEN_5Y_pct_26w",
            "BREAKEVEN_5Y_vs_ma52",  "BREAKEVEN_5Y_above_ma52",
            "BREAKEVEN_5Y_pctile_3y",
            "BREAKEVEN_spread",
            "CPI", "CPI_yoy_pct",
            "CPI_chg_4w", "CPI_chg_8w", "CPI_chg_12w", "CPI_chg_16w", "CPI_chg_26w",
            "CPI_pct_4w", "CPI_pct_8w", "CPI_pct_12w", "CPI_pct_16w", "CPI_pct_26w",
            "CPI_vs_ma52", "CPI_above_ma52", "CPI_pctile_3y",
        ],

        # ----- 3. Fed / tassi nominali -----
        "rates_nominal": [
            "FED_FUNDS",
            "FED_FUNDS_chg_4w",  "FED_FUNDS_chg_8w",
            "FED_FUNDS_chg_12w", "FED_FUNDS_chg_16w", "FED_FUNDS_chg_26w",
            "FED_FUNDS_pct_4w",  "FED_FUNDS_pct_8w",
            "FED_FUNDS_pct_12w", "FED_FUNDS_pct_16w", "FED_FUNDS_pct_26w",
            "FED_FUNDS_vs_ma52", "FED_FUNDS_above_ma52",
            "FED_FUNDS_pctile_3y",
            "DFF",
            "DFF_chg_4w",  "DFF_chg_8w",  "DFF_chg_12w",  "DFF_chg_16w",  "DFF_chg_26w",
            "DFF_pct_4w",  "DFF_pct_8w",  "DFF_pct_12w",  "DFF_pct_16w",  "DFF_pct_26w",
            "DFF_vs_ma52", "DFF_above_ma52", "DFF_pctile_3y",
            "TNX",
            "TNX_chg_4w",  "TNX_chg_8w",  "TNX_chg_12w",  "TNX_chg_16w",  "TNX_chg_26w",
            "TNX_pct_4w",  "TNX_pct_8w",  "TNX_pct_12w",  "TNX_pct_16w",  "TNX_pct_26w",
            "TNX_vs_ma52", "TNX_above_ma52", "TNX_pctile_3y",
            "TLT",
            "TLT_chg_4w",  "TLT_chg_8w",  "TLT_chg_12w",  "TLT_chg_16w",  "TLT_chg_26w",
            "TLT_pct_4w",  "TLT_pct_8w",  "TLT_pct_12w",  "TLT_pct_16w",  "TLT_pct_26w",
            "TLT_vs_ma52", "TLT_above_ma52", "TLT_pctile_3y",
        ],

        # ----- 4. Dollaro USA -----
        "dxy": [
            "DXY",
            "DXY_chg_4w",  "DXY_chg_8w",  "DXY_chg_12w", "DXY_chg_16w", "DXY_chg_26w",
            "DXY_pct_4w",  "DXY_pct_8w",  "DXY_pct_12w", "DXY_pct_16w", "DXY_pct_26w",
            "DXY_vs_ma52", "DXY_above_ma52", "DXY_pctile_3y",
        ],

        # ----- 5. Risk sentiment / volatilità -----
        "risk_sentiment": [
            "VIX",
            "VIX_chg_4w",  "VIX_chg_8w",  "VIX_chg_12w", "VIX_chg_16w", "VIX_chg_26w",
            "VIX_pct_4w",  "VIX_pct_8w",  "VIX_pct_12w", "VIX_pct_16w", "VIX_pct_26w",
            "VIX_vs_ma52", "VIX_above_ma52", "VIX_pctile_3y",
            "VIX_above_20", "VIX_above_30", "VIX_regime", "VIX_chg_4w",
            "EPU",
            "EPU_chg_4w",  "EPU_chg_8w",  "EPU_chg_12w", "EPU_chg_16w", "EPU_chg_26w",
            "EPU_pct_4w",  "EPU_pct_8w",  "EPU_pct_12w", "EPU_pct_16w", "EPU_pct_26w",
            "EPU_vs_ma52", "EPU_above_ma52", "EPU_pctile_3y",
        ],

        # ----- 6. Posizionamento speculatori (COT) -----
        "cot": [
            "COT_OI",
            "COT_OI_chg_4w",  "COT_OI_chg_8w",  "COT_OI_chg_12w", "COT_OI_chg_16w",
            "COT_OI_pct_4w",  "COT_OI_pct_8w",  "COT_OI_pct_12w", "COT_OI_pct_16w",
            "COT_OI_vs_ma52", "COT_OI_pctile_3y",
            "COT_NC_LONG",  "COT_NC_SHORT",
            "COT_net_position",
            "COT_net_chg_4w", "COT_net_chg_8w", "COT_net_chg_12w", "COT_net_chg_16w",
            "COT_net_pct_oi",
            "COT_net_pctile_3y",
            "COT_long_short_ratio",
        ],

        # ----- 7. Geopolitica -----
        "geopolitics": [
            "GPR",
            "GPR_chg_4w",  "GPR_chg_8w",  "GPR_chg_12w", "GPR_chg_16w", "GPR_chg_26w",
            "GPR_pct_4w",  "GPR_pct_8w",  "GPR_pct_12w", "GPR_pct_16w", "GPR_pct_26w",
            "GPR_vs_ma52", "GPR_above_ma52", "GPR_pctile_3y",
            "GPR_ACT",
            "GPR_ACT_chg_4w", "GPR_ACT_chg_8w", "GPR_ACT_chg_12w", "GPR_ACT_chg_16w",
            "GPR_ACT_pct_4w", "GPR_ACT_pct_8w", "GPR_ACT_pct_12w", "GPR_ACT_pct_16w",
            "GPR_ACT_vs_ma52", "GPR_ACT_pctile_3y",
            "GPR_THREAT",
            "GPR_THREAT_chg_4w", "GPR_THREAT_chg_8w", "GPR_THREAT_chg_12w", "GPR_THREAT_chg_16w",
            "GPR_THREAT_pct_4w", "GPR_THREAT_pct_8w", "GPR_THREAT_pct_12w", "GPR_THREAT_pct_16w",
            "GPR_THREAT_vs_ma52", "GPR_THREAT_pctile_3y",
        ],

        # ----- 8. Struttura domanda oro (WGC) -----
        "wgc_structural": [
            "WGC_DEMAND",  "WGC_INVEST",  "WGC_ETF",  "WGC_CB",
            "WGC_DEMAND_chg_4w",  "WGC_INVEST_chg_4w",
            "WGC_ETF_chg_4w",     "WGC_CB_chg_4w",
            "WGC_DEMAND_pct_4w",  "WGC_INVEST_pct_4w",
            "WGC_ETF_pct_4w",     "WGC_CB_pct_4w",
            "WGC_DEMAND_vs_ma52", "WGC_INVEST_vs_ma52",
            "WGC_ETF_vs_ma52",    "WGC_CB_vs_ma52",
            "WGC_DEMAND_pctile_3y","WGC_INVEST_pctile_3y",
            "WGC_ETF_pctile_3y",   "WGC_CB_pctile_3y",
        ],

        # ----- 9. Prezzo oro (proxy momentum) -----
        "gold_momentum": [
            "GOLD",
            "GOLD_chg_4w",  "GOLD_chg_8w",  "GOLD_chg_12w", "GOLD_chg_16w", "GOLD_chg_26w",
            "GOLD_pct_4w",  "GOLD_pct_8w",  "GOLD_pct_12w", "GOLD_pct_16w", "GOLD_pct_26w",
            "GOLD_vs_ma52", "GOLD_above_ma52", "GOLD_pctile_3y",
            "GLD",
            "GLD_chg_4w",   "GLD_chg_8w",   "GLD_chg_12w",  "GLD_chg_16w",  "GLD_chg_26w",
            "GLD_pct_4w",   "GLD_pct_8w",   "GLD_pct_12w",  "GLD_pct_16w",  "GLD_pct_26w",
            "GLD_vs_ma52",  "GLD_above_ma52", "GLD_pctile_3y",
            "IAU",
            "IAU_pct_4w",   "IAU_pct_8w",   "IAU_pct_12w",  "IAU_pct_16w",
            "IAU_vs_ma52",
            "GLD_volume_ma4", "GLD_volume_vs_ma26",
        ],
    }

    # Feature speciali che vogliamo sempre includere (override filtri)
    always_include = {
        "REAL_YIELD_falling_4w",
        "REAL_YIELD_falling_8w",
        "VIX_above_20",
        "VIX_above_30",
        "VIX_regime",
        "COT_net_pctile_3y",
        "COT_net_pct_oi",
        "DXY_pct_4w",
        "DXY_pct_12w",
        "BREAKEVEN_spread",
        "CPI_yoy_pct",
        "GLD_volume_vs_ma26",
    }

    return groups, always_include


# ===========================================================================
# UTILITY CORRELAZIONE
# ===========================================================================

def safe_corr(x: pd.Series, y: pd.Series, min_obs: int = MIN_OBS_FOR_CORR) -> float:
    """Correlazione di Pearson robusta: ritorna 0.0 se dati insufficienti."""
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < min_obs:
        return 0.0
    return float(x.loc[common].corr(y.loc[common]))


def pairwise_prune(
    features: list[str],
    df_train: pd.DataFrame,
    target: pd.Series,
    threshold: float,
) -> list[str]:
    """
    Dato un insieme di feature, rimuove quelle ridondanti per correlazione.
    Strategia greedy: ordina per |r con target| desc, poi per ogni feature
    rimuove le successive con |r pairwise| > threshold.

    Ritorna la lista delle feature sopravvissute.
    """
    # Filtra feature presenti nel DataFrame
    features = [f for f in features if f in df_train.columns]
    if not features:
        return []

    # Calcola correlazione con target (ordinamento priorità)
    corr_target = {}
    for f in features:
        corr_target[f] = abs(safe_corr(df_train[f], target))

    # Ordina per |r_target| decrescente
    ordered = sorted(features, key=lambda f: corr_target[f], reverse=True)

    kept = []
    removed = set()

    for feat in ordered:
        if feat in removed:
            continue
        kept.append(feat)
        # Calcola correlazione con tutte le feature non ancora rimosse
        for other in ordered:
            if other in removed or other == feat or other in kept:
                continue
            r = abs(safe_corr(df_train[feat], df_train[other]))
            if r > threshold:
                removed.add(other)

    return kept


# ===========================================================================
# PIPELINE DI SELEZIONE
# ===========================================================================

def select_features(
    df_feat: pd.DataFrame,
    target_col: str = "target_16w",
) -> tuple[list[str], pd.DataFrame]:
    """
    Esegue la pipeline completa di selezione feature.

    Args:
        df_feat: dataset con feature e target
        target_col: colonna target di riferimento per la selezione

    Returns:
        (lista_feature_selezionate, DataFrame report)
    """
    # Definizione gruppi e feature speciali
    groups, always_include = define_feature_groups()

    # Colonne target e gold_fwd da escludere come feature
    target_cols_all = {c for c in df_feat.columns
                       if c.startswith("target_") or c.startswith("gold_fwd_")}

    # -------------------------------------------------------------------------
    # 1. Estrai periodo training
    # -------------------------------------------------------------------------
    train_mask = (df_feat.index >= TRAIN_START) & (df_feat.index <= TRAIN_END)
    df_train   = df_feat.loc[train_mask].copy()
    target_train = df_train[target_col].dropna()
    # Allinea training alle righe dove il target è disponibile
    df_train = df_train.loc[target_train.index]

    n_train = len(df_train)
    print(f"  Training set: {TRAIN_START} → {TRAIN_END} ({n_train} righe, target_16w disponibile)")

    # -------------------------------------------------------------------------
    # 2. Filtro varianza bassa + correlazione calcolabile (sul training set)
    # -------------------------------------------------------------------------
    all_features = [c for c in df_feat.columns if c not in target_cols_all]
    low_variance = set()
    for f in all_features:
        if f not in df_train.columns:
            continue
        s = df_train[f].dropna()
        # Rimuovi feature con pochi dati o varianza nulla
        if len(s) < MIN_OBS_FOR_CORR or s.std() < MIN_STD:
            low_variance.add(f)
            continue
        # Rimuovi feature la cui correlazione con il target risulta NaN
        # (accade quando la serie ha ancora inf/NaN che azzerano la varianza)
        r = safe_corr(df_train[f], target_train)
        if np.isnan(r):
            low_variance.add(f)
    print(f"  Filtro varianza bassa / correlazione incalcolabile: {len(low_variance)} feature rimosse")

    # -------------------------------------------------------------------------
    # 3. Intra-group pruning
    # -------------------------------------------------------------------------
    report_rows = []
    candidates_per_group = {}

    for group_name, group_feats in groups.items():
        # Filtra: presente nel df, non a bassa varianza, non target
        valid = [f for f in group_feats
                 if f in df_train.columns
                 and f not in low_variance
                 and f not in target_cols_all]

        if not valid:
            candidates_per_group[group_name] = []
            continue

        # Calcola correlazioni con target_16w
        for f in valid:
            r = safe_corr(df_train[f], target_train)
            n = df_train[f].dropna().shape[0]
            report_rows.append({
                "feature": f, "group": group_name,
                "pearson_r_target16w": r, "abs_r": abs(r), "n_train": n,
            })

        # Pruning intra-gruppo
        pruned = pairwise_prune(valid, df_train, target_train, INTRA_GROUP_CORR_THRESHOLD)

        # Limita a N_TOP per gruppo, sempre tenendo always_include
        always_in_group = [f for f in pruned if f in always_include]
        others = [f for f in pruned if f not in always_include]
        n_slots = max(0, N_TOP_PER_GROUP - len(always_in_group))
        top_others = sorted(others,
                            key=lambda f: abs(safe_corr(df_train[f], target_train)),
                            reverse=True)[:n_slots]

        group_selected = always_in_group + top_others
        candidates_per_group[group_name] = group_selected

        print(f"  Gruppo '{group_name}':"
              f" {len(valid)} valide → {len(pruned)} post-pruning → {len(group_selected)} selezionate")

    # -------------------------------------------------------------------------
    # 4. Raccolta candidati + cross-group pruning
    # -------------------------------------------------------------------------
    all_candidates = []
    seen = set()
    for g_feats in candidates_per_group.values():
        for f in g_feats:
            if f not in seen:
                all_candidates.append(f)
                seen.add(f)

    # Aggiungi always_include non già presenti
    for f in always_include:
        if f in df_train.columns and f not in seen and f in all_features:
            all_candidates.append(f)
            seen.add(f)

    print(f"\n  Candidati dopo pruning intra-gruppo: {len(all_candidates)}")

    # Cross-group pruning
    final_features = pairwise_prune(
        all_candidates, df_train, target_train, CROSS_GROUP_CORR_THRESHOLD
    )

    # Garantisci always_include che potrebbero essere stati rimossi per
    # correlazione con un'altra feature (re-inserimento forzato)
    forced_back = []
    for f in always_include:
        if f in df_train.columns and f not in final_features and f in all_features:
            forced_back.append(f)
            final_features.append(f)

    if forced_back:
        print(f"  Feature speciali re-inserite forzatamente: {forced_back}")

    print(f"  Feature dopo pruning cross-gruppo: {len(final_features)}")

    # -------------------------------------------------------------------------
    # 5. Build report completo
    # -------------------------------------------------------------------------
    report_df = pd.DataFrame(report_rows)
    if not report_df.empty:
        report_df["selected"] = report_df["feature"].isin(set(final_features))
        report_df = report_df.sort_values("abs_r", ascending=False).reset_index(drop=True)

    return final_features, report_df


# ===========================================================================
# ANALISI TEMATICA
# ===========================================================================

def print_thematic_summary(selected: list[str], report_df: pd.DataFrame) -> None:
    """Stampa riepilogo per gruppo tematico delle feature selezionate."""
    groups, _ = define_feature_groups()

    # Mappa feature → gruppo
    feat_to_group = {}
    for g, feats in groups.items():
        for f in feats:
            feat_to_group[f] = g

    print("\n  Feature selezionate per gruppo tematico:")
    group_counts = {}
    for f in selected:
        g = feat_to_group.get(f, "other")
        group_counts.setdefault(g, []).append(f)

    for g, feats in sorted(group_counts.items()):
        print(f"\n  [{g}] ({len(feats)} feature)")
        for f in sorted(feats):
            r_row = report_df.loc[report_df["feature"] == f, "pearson_r_target16w"]
            r_val = f"{r_row.values[0]:+.4f}" if len(r_row) > 0 else "  n/a "
            print(f"    {f:<45} r={r_val}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 70)
    print("  FACTOR ANALYSIS & SELEZIONE VARIABILI")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Carica dataset features
    features_path = PROCESSED_DIR / "dataset_features.csv"
    if not features_path.exists():
        print(f"\n  ERRORE: {features_path} non trovato.")
        print("  Eseguire prima: python -m src.features.feature_engineering")
        sys.exit(1)

    print(f"\n  Caricamento: {features_path.name}")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)
    print(f"  Shape: {df.shape}")

    all_feat_cols = [c for c in df.columns
                     if not c.startswith("target_") and not c.startswith("gold_fwd_")]
    target_cols   = [c for c in df.columns
                     if c.startswith("target_") or c.startswith("gold_fwd_")]
    print(f"  Feature disponibili: {len(all_feat_cols)}")
    print(f"  Colonne target:      {len(target_cols)}")

    # Esegui selezione
    print("\n  Avvio pipeline di selezione...")
    selected, report_df = select_features(df, target_col="target_16w")

    # Riepilogo tematico
    print_thematic_summary(selected, report_df)

    # -------------------------------------------------------------------------
    # Report finale
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  REPORT FINALE")
    print("=" * 70)
    print(f"\n  Feature originali:         {len(all_feat_cols)}")
    print(f"  Feature selezionate:       {len(selected)}")
    print(f"  Riduzione:                 {100*(1 - len(selected)/len(all_feat_cols)):.1f}%")

    print(f"\n  Top 20 feature selezionate per |r| con target_16w (training 2005-2015):")  
    print(f"  {'#':<4} {'FEATURE':<45} {'r':>7}  {'GRUPPO':<20}")
    print("  " + "-" * 82)
    groups_map, _ = define_feature_groups()
    feat_to_group = {f: g for g, feats in groups_map.items() for f in feats}
    if not report_df.empty:
        sel_report = report_df.loc[report_df["selected"]].sort_values("abs_r", ascending=False)
        for i, (_, row) in enumerate(sel_report.head(20).iterrows()):
            g = feat_to_group.get(row["feature"], "other")
            print(f"  {i+1:<4} {row['feature']:<45} {row['pearson_r_target16w']:>+7.4f}  {g:<20}")

    # -------------------------------------------------------------------------
    # Salvataggio
    # -------------------------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Lista feature selezionate
    sel_list_path = PROCESSED_DIR / "selected_features.txt"
    sel_list_path.write_text("\n".join(selected), encoding="utf-8")

    # 2. Dataset con solo feature selezionate + target
    keep_cols = selected + target_cols
    existing_keep = [c for c in keep_cols if c in df.columns]
    df_selected = df[existing_keep].copy()
    sel_dataset_path = PROCESSED_DIR / "dataset_selected.csv"
    df_selected.to_csv(sel_dataset_path)

    # 3. Report correlazioni
    if not report_df.empty:
        report_path = PROCESSED_DIR / "factor_analysis_report.csv"
        report_df.to_csv(report_path, index=False)
        print(f"\n  ✓ Salvato: {report_path.relative_to(PROJECT_ROOT)}")

    print(f"  ✓ Salvato: {sel_list_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Salvato: {sel_dataset_path.relative_to(PROJECT_ROOT)}")
    print(f"    {df_selected.shape[0]} righe × {df_selected.shape[1]} colonne"
          f" ({len(selected)} feature + {len(target_cols)} target)")

    # Verifica distribuzione NaN sulle feature selezionate (periodo 2005-2025)
    df_check = df_selected.loc["2005":"2025", selected]
    nan_summary = (df_check.isna().mean() * 100).sort_values(ascending=False)
    heavy = nan_summary[nan_summary > 30]
    if len(heavy) > 0:
        print(f"\n  WARNING — Feature selezionate con >30% NaN (2005-2025):")
        for feat, pct in heavy.items():
            print(f"    {feat}: {pct:.1f}%")
        print("  (LightGBM gestisce NaN nativamente — accettabile)")
    else:
        print(f"\n  ✓ Nessuna feature selezionata con >30% NaN nel periodo 2005-2025")

    print()


if __name__ == "__main__":
    main()
