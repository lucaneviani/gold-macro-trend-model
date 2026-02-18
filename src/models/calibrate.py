"""
calibrate.py — Calibrazione Platt + score composito 0-100

Esecuzione:
    cd gold_model
    python -m src.models.calibrate

COSA FA:
    1. Esegue walk-forward OOS per tutti e 3 i target (12w, 16w, 26w)
    2. Calibra le probabilità raw con Platt scaling (LogisticRegression)
       fit SOLO sulle predizioni OOS → nessun lookahead
    3. Combina i 3 target in uno score composito 0-100:
           score = 100 × (0.25 × p12_cal + 0.50 × p16_cal + 0.25 × p26_cal)
    4. Classi finali: LONG (score > 65), FLAT (35-65), SHORT (< 35)
    5. Addestra modelli finali su TUTTI i dati disponibili
    6. Calcola il segnale corrente calibrato

OUTPUT:
    outputs/results/calibrated_predictions.csv  — OOS predizioni calibrate
    outputs/results/current_score.csv           — segnale finale settimana corrente
    outputs/results/calibration_report.txt      — report testuale
    models/lgbm_12w.pkl / lgbm_16w.pkl / lgbm_26w.pkl  — modelli finali
    models/platt_12w.pkl / platt_16w.pkl / platt_26w.pkl — calibratori

NOTA ANTI-LOOKAHEAD CALIBRAZIONE:
    Il calibratore Platt è fittato sulle predizioni OOS del walk-forward
    (2016-2025), che sono dati il modello non ha mai visto durante il training.
    Le probabilità calibrate per la settimana corrente usano i calibratori
    fittati sull'intero periodo OOS (2016-2025).
"""

import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from io import StringIO

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
    LGBM_PARAMS_BASE,
    WF_TRAIN_START,
    WF_FIRST_TEST,
    SCORE_LONG_THRESHOLD,
    SCORE_SHORT_THRESHOLD,
)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, brier_score_loss
except ImportError as e:
    print(f"ERRORE import sklearn: {e}")
    sys.exit(1)

# Importa utility da model.py
from src.models.model import load_data, get_fold_dates, train_lgbm

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
MODELS_DIR         = PROJECT_ROOT / "models"
TARGETS_ALL        = ["target_12w", "target_16w", "target_26w"]
WEIGHTS            = {"target_12w": 0.25, "target_16w": 0.50, "target_26w": 0.25}
HORIZON_LABEL      = {"target_12w": "12w", "target_16w": "16w", "target_26w": "26w"}


# ===========================================================================
# WALK-FORWARD MULTI-TARGET
# ===========================================================================

def run_wf_for_target(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """
    Esegue il walk-forward per un singolo target.
    Ritorna DataFrame con colonne: date, prob_raw, actual_target, fold.
    """
    folds = get_fold_dates(df, WF_FIRST_TEST)
    rows  = []

    for train_end, test_start, test_end in folds:
        train_mask = (df.index >= WF_TRAIN_START) & (df.index <= train_end)
        df_train   = df.loc[train_mask].dropna(subset=[target])
        if len(df_train) < 80:
            continue

        X_train = df_train[features]
        y_train = df_train[target].astype(int)
        model   = train_lgbm(X_train, y_train, LGBM_PARAMS_BASE)

        test_mask = (df.index >= test_start) & (df.index <= test_end)
        df_test   = df.loc[test_mask].dropna(subset=[target])
        if len(df_test) < 5:
            continue

        probs = model.predict_proba(df_test[features])[:, 1]
        for dt, prob, actual in zip(df_test.index, probs, df_test[target].values):
            rows.append({
                "date":          dt,
                f"prob_raw_{HORIZON_LABEL[target]}": prob,
                f"actual_{HORIZON_LABEL[target]}":   actual,
                "fold":          test_start[:4],
            })

    return pd.DataFrame(rows).set_index("date")


# ===========================================================================
# PLATT SCALING
# ===========================================================================

def fit_platt(
    prob_raw: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:
    """
    Fit Platt scaling: LogisticRegression con C=1 su predizioni OOS.
    Input: prob_raw (N,), y_true (N,)
    Output: calibratore fittato (usa .predict_proba(prob.reshape(-1,1)))
    """
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    platt.fit(prob_raw.reshape(-1, 1), y_true)
    return platt


def apply_platt(
    platt: LogisticRegression,
    prob_raw: float | np.ndarray,
) -> float | np.ndarray:
    """Applica il calibratore Platt a una o più probabilità raw."""
    scalar = np.isscalar(prob_raw)
    x = np.atleast_1d(float(prob_raw) if scalar else prob_raw).reshape(-1, 1)
    cal = platt.predict_proba(x)[:, 1]
    return float(cal[0]) if scalar else cal


# ===========================================================================
# CALIBRATION DIAGNOSTICS
# ===========================================================================

def calibration_stats(
    y_true: np.ndarray,
    prob_raw: np.ndarray,
    prob_cal: np.ndarray,
    label: str,
) -> dict:
    """Calcola statistiche di calibrazione prima/dopo."""
    n_pos = int(y_true.sum())
    n     = len(y_true)

    def _stats(p: np.ndarray, name: str) -> dict:
        try:
            auc = roc_auc_score(y_true, p)
        except Exception:
            auc = np.nan
        brier = brier_score_loss(y_true, p)
        # Expected Calibration Error (ECE) con 10 bin
        try:
            frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
            ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        except Exception:
            ece = np.nan
        return {f"{name}_auc": round(auc, 4), f"{name}_brier": round(brier, 4),
                f"{name}_ece": round(ece, 4)}

    result = {"target": label, "n_obs": n, "pct_pos": round(n_pos / n * 100, 1)}
    result.update(_stats(prob_raw, "raw"))
    result.update(_stats(prob_cal, "cal"))
    return result


# ===========================================================================
# SCORE COMPOSITO
# ===========================================================================

def compute_score(probs_cal: dict[str, float]) -> dict:
    """
    Calcola lo score composito 0-100 dai 3 target calibrati.

    probs_cal: {"target_12w": p12, "target_16w": p16, "target_26w": p26}
    """
    # Composito pesato
    composite = sum(WEIGHTS[t] * probs_cal.get(t, np.nan) for t in TARGETS_ALL)

    # Gestisci NaN (se un target manca usa solo quelli disponibili)
    valid_targets = [t for t in TARGETS_ALL if not np.isnan(probs_cal.get(t, np.nan))]
    if not valid_targets:
        composite = np.nan
    elif len(valid_targets) < len(TARGETS_ALL):
        total_w = sum(WEIGHTS[t] for t in valid_targets)
        composite = sum(WEIGHTS[t] * probs_cal[t] for t in valid_targets) / total_w

    score = round(composite * 100, 1) if not np.isnan(composite) else np.nan

    # Classificazione
    if np.isnan(score):
        signal = "UNDEFINED"
    elif score >= SCORE_LONG_THRESHOLD:
        signal = "LONG"
    elif score <= SCORE_SHORT_THRESHOLD:
        signal = "SHORT"
    else:
        signal = "FLAT"

    return {
        "score":      score,
        "signal":     signal,
        "composite_prob": round(composite, 4) if not np.isnan(composite) else np.nan,
        **{f"prob_cal_{HORIZON_LABEL[t]}": round(probs_cal.get(t, np.nan), 4)
           for t in TARGETS_ALL},
    }


# ===========================================================================
# MODELLO FINALE (addestrato su tutti i dati)
# ===========================================================================

def train_final_models(
    df: pd.DataFrame,
    features: list[str],
    platts: dict[str, LogisticRegression],
) -> tuple[dict, pd.DataFrame]:
    """
    Addestra i modelli finali su tutti i dati disponibili.
    Ritorna (dict target→model, DataFrame segnale corrente).
    """
    final_models = {}
    last_row     = df.iloc[[-1]]
    X_last       = last_row[features]
    probs_cal    = {}

    for target in TARGETS_ALL:
        df_t = df.loc[df[target].notna()].copy()
        if len(df_t) < 50:
            continue

        model = train_lgbm(df_t[features], df_t[target].astype(int), LGBM_PARAMS_BASE)
        final_models[target] = model

        prob_raw = float(model.predict_proba(X_last)[:, 1][0])
        platt    = platts.get(target)
        prob_c   = apply_platt(platt, prob_raw) if platt is not None else prob_raw
        probs_cal[target] = float(prob_c)

        h = HORIZON_LABEL[target]
        print(f"    {target}: prob_raw={prob_raw:.4f}  →  prob_cal={prob_c:.4f}")

    signal_data = compute_score(probs_cal)
    signal_data["date"]         = last_row.index[0]
    signal_data["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    # aggiungi raw probs per trasparenza
    for target in TARGETS_ALL:
        if target in final_models:
            prob_raw = float(final_models[target].predict_proba(X_last)[:, 1][0])
            signal_data[f"prob_raw_{HORIZON_LABEL[target]}"] = round(prob_raw, 4)

    return final_models, pd.DataFrame([signal_data])


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 72)
    print("  CALIBRAZIONE PLATT + SCORE COMPOSITO 0-100")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Carica dati
    print("\n  Caricamento dataset...")
    df, features = load_data()
    print(f"  Dataset: {df.shape[0]} righe × {df.shape[1]} colonne")
    print(f"  Feature: {len(features)}")

    # ------------------------------------------------------------------ #
    # 1. Walk-forward OOS per tutti e 3 i target
    # ------------------------------------------------------------------ #
    print(f"\n  Walk-forward OOS per 3 target ({WF_FIRST_TEST[:4]}–2025)...")
    wf_results = {}
    for target in TARGETS_ALL:
        h = HORIZON_LABEL[target]
        print(f"    Target {target}...", end=" ", flush=True)
        wf_results[target] = run_wf_for_target(df, features, target)
        n = len(wf_results[target])
        print(f"{n} predizioni OOS")

    # Unisci tutte le OOS predictions in un unico DataFrame
    oos_dfs = []
    for target, wdf in wf_results.items():
        oos_dfs.append(wdf)
    combined_oos = pd.concat(oos_dfs, axis=1)

    # ------------------------------------------------------------------ #
    # 2. Platt scaling per ciascun target
    # ------------------------------------------------------------------ #
    print(f"\n  Calibrazione Platt scaling...")
    platts     = {}
    cal_stats  = []

    for target in TARGETS_ALL:
        h   = HORIZON_LABEL[target]
        wdf = wf_results[target]
        if wdf.empty:
            print(f"    {target}: SKIP (nessun dato OOS)")
            continue

        prob_raw_col   = f"prob_raw_{h}"
        actual_col     = f"actual_{h}"

        if prob_raw_col not in wdf.columns or actual_col not in wdf.columns:
            print(f"    {target}: SKIP (colonne mancanti)")
            continue

        mask     = wdf[actual_col].notna() & wdf[prob_raw_col].notna()
        prob_raw = wdf.loc[mask, prob_raw_col].values
        y_true   = wdf.loc[mask, actual_col].values.astype(int)

        if len(prob_raw) < 30:
            print(f"    {target}: SKIP (n={len(prob_raw)} < 30)")
            continue

        platt      = fit_platt(prob_raw, y_true)
        prob_cal   = apply_platt(platt, prob_raw)
        platts[target] = platt

        stats = calibration_stats(y_true, prob_raw, prob_cal, h)
        cal_stats.append(stats)

        print(
            f"    {target}: n={stats['n_obs']} ({stats['pct_pos']}% pos)  "
            f"AUC: raw={stats['raw_auc']:.4f}→cal={stats['cal_auc']:.4f}  "
            f"ECE: raw={stats['raw_ece']:.4f}→cal={stats['cal_ece']:.4f}  "
            f"Brier: raw={stats['raw_brier']:.4f}→cal={stats['cal_brier']:.4f}"
        )

        # Aggiorna combinata con probs calibrate
        wf_results[target][f"prob_cal_{h}"] = np.nan
        wf_results[target].loc[mask, f"prob_cal_{h}"] = prob_cal

    # ------------------------------------------------------------------ #
    # 3. Score composito OOS (per ogni settimana che ha tutti e 3 i target)
    # ------------------------------------------------------------------ #
    print(f"\n  Calcolo score composito OOS...")

    # Ricostruisci DataFrame combinato con probs calibrate
    col_map = {}
    for target in TARGETS_ALL:
        h = HORIZON_LABEL[target]
        wdf = wf_results.get(target, pd.DataFrame())
        if f"prob_cal_{h}" in wdf.columns:
            col_map[f"prob_cal_{h}"]    = wdf[f"prob_cal_{h}"]
            col_map[f"actual_{h}"]      = wdf.get(f"actual_{h}", pd.Series(dtype=float))

    if col_map:
        oos_all = pd.DataFrame(col_map)
    else:
        oos_all = pd.DataFrame()

    if not oos_all.empty:
        scores = []
        for dt, row in oos_all.iterrows():
            p = {t: row.get(f"prob_cal_{HORIZON_LABEL[t]}", np.nan) for t in TARGETS_ALL}
            s = compute_score(p)
            s["date"] = dt
            scores.append(s)
        scores_df = pd.DataFrame(scores).set_index("date")

        # Distribuzione segnale OOS
        sig_counts = scores_df["signal"].value_counts()
        print(f"  Distribuzione segnale OOS (2016-2025):")
        for sig in ["LONG", "FLAT", "SHORT"]:
            n = sig_counts.get(sig, 0)
            tot = len(scores_df)
            print(f"    {sig:<5}: {n:3d} ({n/tot*100:.1f}%)")

        # Accuracy del segnale LONG vs actual target_16w
        if "actual_16w" in oos_all.columns and "signal" in scores_df.columns:
            merged    = oos_all["actual_16w"].dropna().rename("actual_16w")
            sig_check = scores_df["signal"].reindex(merged.index)
            n_long    = (sig_check == "LONG").sum()
            n_long_correct = ((sig_check == "LONG") & (merged == 1)).sum()
            n_short   = (sig_check == "SHORT").sum()
            n_short_correct = ((sig_check == "SHORT") & (merged == 0)).sum()

            if n_long > 0:
                print(f"\n  Accuratezza segnali direzionali OOS:")
                print(f"    LONG signals : {n_long:3d}  →  corretto {n_long_correct}/{n_long} ({n_long_correct/n_long*100:.1f}%)")
            if n_short > 0:
                print(f"    SHORT signals: {n_short:3d}  →  corretto {n_short_correct}/{n_short} ({n_short_correct/n_short*100:.1f}%)")

    # ------------------------------------------------------------------ #
    # 4. Modelli finali + segnale corrente
    # ------------------------------------------------------------------ #
    print(f"\n  Addestramento modelli finali (tutti i dati disponibili)...")
    final_models, current_df = train_final_models(df, features, platts)

    score_val  = current_df["score"].values[0]
    signal_val = current_df["signal"].values[0]
    date_val   = current_df["date"].values[0]

    print(f"\n  {'='*40}")
    print(f"  SEGNALE CORRENTE — {pd.Timestamp(date_val).strftime('%Y-%m-%d')}")
    print(f"  {'='*40}")
    print(f"  Score composito:  {score_val:.1f} / 100")
    print(f"  Segnale:          {signal_val}")
    print(f"  Prob calibrata 12w:  {current_df['prob_cal_12w'].values[0]:.4f}")
    print(f"  Prob calibrata 16w:  {current_df['prob_cal_16w'].values[0]:.4f}")
    print(f"  Prob calibrata 26w: {current_df['prob_cal_26w'].values[0]:.4f}")
    print(f"\n  Soglie: LONG > {SCORE_LONG_THRESHOLD}, SHORT < {SCORE_SHORT_THRESHOLD}")
    print(f"  {'='*40}")

    # ------------------------------------------------------------------ #
    # 5. Salvataggio
    # ------------------------------------------------------------------ #
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Predizioni calibrate OOS
    if not oos_all.empty and "scores_df" in dir():
        cal_pred_path = RESULTS_DIR / "calibrated_predictions.csv"
        oos_all.join(scores_df[["score", "signal", "composite_prob"]]).to_csv(cal_pred_path)
        print(f"\n  ✓ Salvato: {cal_pred_path.relative_to(PROJECT_ROOT)}")

    # Segnale corrente
    score_path = RESULTS_DIR / "current_score.csv"
    current_df.to_csv(score_path, index=False)
    print(f"  ✓ Salvato: {score_path.relative_to(PROJECT_ROOT)}")

    # Modelli finali e calibratori
    for target, model in final_models.items():
        h = HORIZON_LABEL[target]
        mdl_path = MODELS_DIR / f"lgbm_{h}.pkl"
        with open(mdl_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  ✓ Salvato: {mdl_path.relative_to(PROJECT_ROOT)}")

    for target, platt in platts.items():
        h = HORIZON_LABEL[target]
        plt_path = MODELS_DIR / f"platt_{h}.pkl"
        with open(plt_path, "wb") as f:
            pickle.dump(platt, f)
        print(f"  ✓ Salvato: {plt_path.relative_to(PROJECT_ROOT)}")

    # Report testuale
    report_lines = [
        "=" * 60,
        "  CALIBRATION REPORT — Gold Trend Prediction Model",
        f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "  STATISTICHE CALIBRAZIONE",
        "",
    ]
    for cs in cal_stats:
        report_lines += [
            f"  Target {cs['target']}w:  n={cs['n_obs']} ({cs['pct_pos']}% positivi)",
            f"    AUC:   raw={cs['raw_auc']:.4f} → cal={cs['cal_auc']:.4f}",
            f"    Brier: raw={cs['raw_brier']:.4f} → cal={cs['cal_brier']:.4f}",
            f"    ECE:   raw={cs['raw_ece']:.4f} → cal={cs['cal_ece']:.4f}",
            "",
        ]
    report_lines += [
        "  SEGNALE CORRENTE",
        f"  Data:             {pd.Timestamp(date_val).strftime('%Y-%m-%d')}",
        f"  Score:            {score_val:.1f} / 100",
        f"  Segnale:          {signal_val}",
        f"  Prob cal 12w:      {current_df['prob_cal_12w'].values[0]:.4f}",
        f"  Prob cal 16w:      {current_df['prob_cal_16w'].values[0]:.4f}",
        f"  Prob cal 26w:      {current_df['prob_cal_26w'].values[0]:.4f}",
        f"  Prob raw 12w:      {current_df.get('prob_raw_12w', pd.Series([np.nan])).values[0]:.4f}",
        f"  Prob raw 16w:      {current_df.get('prob_raw_16w', pd.Series([np.nan])).values[0]:.4f}",
        f"  Prob raw 26w:      {current_df.get('prob_raw_26w', pd.Series([np.nan])).values[0]:.4f}",
        "",
        "  PESI COMBINAZIONE",
        f"  target_12w: {WEIGHTS['target_12w']*100:.0f}%",
        f"  target_16w: {WEIGHTS['target_16w']*100:.0f}% (target principale)",
        f"  target_26w: {WEIGHTS['target_26w']*100:.0f}%",
    ]

    report_path = RESULTS_DIR / "calibration_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  ✓ Salvato: {report_path.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
