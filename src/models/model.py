"""
model.py — Walk-forward LightGBM ensemble per previsione trend oro

Esecuzione:
    cd gold_model
    python -m src.models.model

STRATEGIA WALK-FORWARD (expanding window, no lookahead):
    Fold 1  — train: 2005-01-01 → 2015-12-31  |  test: 2016-01-01 → 2016-12-31
    Fold 2  — train: 2005-01-01 → 2016-12-31  |  test: 2017-01-01 → 2017-12-31
    ...
    Fold N  — train: 2005-01-01 → (anno-1)-12-31 | test: anno-01-01 → anno-12-31
    Current — train: 2005-01-01 → last available  | predict: ultima settimana disponibile

TARGET PRINCIPALE: target_16w (binario — oro +≥2% in 16 settimane)
TARGET SECONDARI:  target_12w, target_26w (stessa logica, orizzonte diverso)

OUTPUT:
    outputs/results/wf_predictions.csv      — predizioni out-of-sample + actual
    outputs/results/wf_metrics.csv          — metriche per fold (AUC, logloss, ecc.)
    outputs/results/feature_importance.csv  — importanza media feature su tutti i fold
    outputs/results/current_signal.csv      — segnale corrente (ultima settimana)
    models/lgbm_current.pkl                 — modello addestrato su tutti i dati
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
    LGBM_PARAMS_BASE,
    WF_TRAIN_START,
    WF_FIRST_TEST,
)

try:
    import lightgbm as lgb
    from sklearn.metrics import (
        roc_auc_score, log_loss, accuracy_score,
        precision_score, recall_score, f1_score, brier_score_loss,
    )
except ImportError as e:
    print(f"ERRORE import: {e}")
    print("Eseguire: C:/Python314/python.exe -m pip install lightgbm scikit-learn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
MODELS_DIR     = PROJECT_ROOT / "models"
TARGETS_ALL    = ["target_12w", "target_16w", "target_26w"]
TARGET_PRIMARY = "target_16w"


# ===========================================================================
# UTILITY
# ===========================================================================

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Carica dataset_selected.csv e selected_features.txt.
    Ritorna (DataFrame completo, lista feature selezionate).
    """
    ds_path = PROCESSED_DIR / "dataset_selected.csv"
    sf_path = PROCESSED_DIR / "selected_features.txt"

    if not ds_path.exists():
        raise FileNotFoundError(f"{ds_path} non trovato. Eseguire prima factor_analysis.py")
    if not sf_path.exists():
        raise FileNotFoundError(f"{sf_path} non trovato. Eseguire prima factor_analysis.py")

    df = pd.read_csv(ds_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=False).tz_localize(None)

    features = sf_path.read_text(encoding="utf-8").strip().splitlines()
    # Tieni solo feature effettivamente presenti nel dataset
    features = [f for f in features if f in df.columns]

    return df, features


def get_fold_dates(df: pd.DataFrame, first_test: str) -> list[tuple[str, str, str]]:
    """
    Genera la lista di fold per il walk-forward.

    Ritorna lista di (train_end, test_start, test_end) per ogni anno di test.
    L'ultimo fold copre fino all'ultimo anno disponibile con dati target validi.
    """
    first_test_dt  = pd.Timestamp(first_test)
    # Anno massimo per cui abbiamo target_16w: l'ultimo anno "chiuso" nel dataset
    # (le ultime 16 settimane non hanno target → escludiamo l'anno in corso se
    #  non è completamente chiuso)
    last_valid_target = df[TARGET_PRIMARY].last_valid_index()
    last_test_year    = last_valid_target.year

    folds = []
    for test_year in range(first_test_dt.year, last_test_year + 1):
        train_end  = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end   = f"{test_year}-12-31"
        folds.append((train_end, test_start, test_end))

    return folds


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
) -> lgb.LGBMClassifier:
    """
    Addestra un LGBMClassifier con i parametri forniti.
    Gestisce il class imbalance automaticamente.
    """
    # Calcola peso classi per bilanciamento (utile quando yr ha pochi campioni)
    n_pos  = y_train.sum()
    n_neg  = len(y_train) - n_pos
    scale  = n_neg / n_pos if n_pos > 0 else 1.0

    p = {**params, "scale_pos_weight": scale}
    model = lgb.LGBMClassifier(**p)
    model.fit(X_train, y_train)
    return model


def compute_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    """Calcola il set completo di metriche su un fold."""
    y_pred = (y_prob >= threshold).astype(int)
    n      = len(y_true)
    n_pos  = int(y_true.sum())

    metrics = {
        "fold":          label,
        "n_test":        n,
        "n_positive":    n_pos,
        "pct_positive":  round(n_pos / n * 100, 1),
        "auc":           round(roc_auc_score(y_true, y_prob), 4)    if n_pos > 0 and n_pos < n else np.nan,
        "logloss":       round(log_loss(y_true, y_prob), 4),
        "brier":         round(brier_score_loss(y_true, y_prob), 4),
        "accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    return metrics


# ===========================================================================
# WALK-FORWARD PRINCIPALE
# ===========================================================================

def run_walk_forward(
    df: pd.DataFrame,
    features: list[str],
    target: str = TARGET_PRIMARY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Esegue il walk-forward completo.

    Ritorna:
        predictions_df  — tutte le predizioni out-of-sample
        metrics_df      — metriche per fold
        importance_df   — importanza feature media su tutti i fold
    """
    folds = get_fold_dates(df, WF_FIRST_TEST)
    print(f"  Fold pianificati: {len(folds)}")
    print(f"  Primo fold: train fino al {folds[0][0]}, test {folds[0][1]} → {folds[0][2]}")
    print(f"  Ultimo fold: train fino al {folds[-1][0]}, test {folds[-1][1]} → {folds[-1][2]}\n")

    all_preds   = []
    all_metrics = []
    importance_accumulator: dict[str, list[float]] = {f: [] for f in features}

    for i, (train_end, test_start, test_end) in enumerate(folds):
        # ----------------------------------------------------------------- #
        # Dati di training
        # ----------------------------------------------------------------- #
        train_mask = (df.index >= WF_TRAIN_START) & (df.index <= train_end)
        df_train   = df.loc[train_mask].copy()

        # Rimuovi righe dove il target è NaN nel training
        df_train = df_train.loc[df_train[target].notna()]

        X_train = df_train[features]
        y_train = df_train[target].astype(int)

        # ----------------------------------------------------------------- #
        # Dati di test
        # ----------------------------------------------------------------- #
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        df_test   = df.loc[test_mask].copy()

        # Tieni solo righe con target valido per le metriche
        df_test_eval = df_test.loc[df_test[target].notna()].copy()

        if len(df_train) < 100 or len(df_test_eval) < 5:
            print(f"  [Fold {i+1:02d}] {test_start[:4]} — SKIP (dati insufficienti)")
            continue

        # ----------------------------------------------------------------- #
        # Training
        # ----------------------------------------------------------------- #
        model = train_lgbm(X_train, y_train, LGBM_PARAMS_BASE)

        # ----------------------------------------------------------------- #
        # Predizioni sul test set (incluse righe senza target per completezza)
        # ----------------------------------------------------------------- #
        X_test_all = df_test[features]
        prob_all   = model.predict_proba(X_test_all)[:, 1]

        X_test_eval = df_test_eval[features]
        prob_eval   = model.predict_proba(X_test_eval)[:, 1]
        y_eval      = df_test_eval[target].astype(int)

        # ----------------------------------------------------------------- #
        # Metriche
        # ----------------------------------------------------------------- #
        metrics = compute_metrics(
            y_eval, prob_eval,
            label=test_start[:4],
        )
        metrics["train_rows"] = len(df_train)
        metrics["train_end"]  = train_end
        all_metrics.append(metrics)

        # ----------------------------------------------------------------- #
        # Salva predizioni out-of-sample (con target, se disponibile)
        # ----------------------------------------------------------------- #
        pred_df = pd.DataFrame({
            "date":          df_test.index,
            "prob_lgbm_16w":  prob_all,
            "actual_target": df_test[target].values,
            "gold_fwd_16w":   df_test.get("gold_fwd_16w", pd.Series(dtype=float)).values,
            "fold":          test_start[:4],
        })
        all_preds.append(pred_df)

        # ----------------------------------------------------------------- #
        # Feature importance
        # ----------------------------------------------------------------- #
        fi = dict(zip(features, model.feature_importances_))
        for f in features:
            importance_accumulator[f].append(fi.get(f, 0))

        auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics['auc']) else " n/a "
        print(
            f"  [Fold {i+1:02d}] {test_start[:4]}"
            f"  train_rows={metrics['train_rows']:4d}"
            f"  test_rows={metrics['n_test']:3d}"
            f"  AUC={auc_str}"
            f"  logloss={metrics['logloss']:.4f}"
            f"  precision={metrics['precision']:.3f}"
            f"  recall={metrics['recall']:.3f}"
        )

    # ------------------------------------------------------------------ #
    # Aggrega output
    # ------------------------------------------------------------------ #
    predictions_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    metrics_df     = pd.DataFrame(all_metrics)

    # Importanza media su tutti i fold
    importance_df = pd.DataFrame({
        "feature":       features,
        "importance_mean": [np.mean(importance_accumulator[f]) for f in features],
        "importance_std":  [np.std(importance_accumulator[f])  for f in features],
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return predictions_df, metrics_df, importance_df


# ===========================================================================
# MODELLO CORRENTE (predizione ultima settimana disponibile)
# ===========================================================================

def train_current_model(
    df: pd.DataFrame,
    features: list[str],
    target: str = TARGET_PRIMARY,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    """
    Addestra il modello su TUTTI i dati disponibili con target valido,
    poi predice l'ultima settimana disponibile nel dataset.

    Questo è il modello che viene usato per il segnale corrente.
    """
    df_train = df.loc[df[target].notna()].copy()
    X_train  = df_train[features]
    y_train  = df_train[target].astype(int)

    model = train_lgbm(X_train, y_train, LGBM_PARAMS_BASE)

    # Ultima riga disponibile (incluse settimane senza target ancora confermato)
    last_row = df.iloc[[-1]]
    X_last   = last_row[features]
    prob_8w  = float(model.predict_proba(X_last)[:, 1][0])

    # Allena anche modelli secondari per 12w e 26w
    secondary_probs = {}
    for t in ["target_12w", "target_26w"]:
        if t in df.columns:
            df_t   = df.loc[df[t].notna()].copy()
            m_sec  = train_lgbm(df_t[features], df_t[t].astype(int), LGBM_PARAMS_BASE)
            secondary_probs[t] = float(m_sec.predict_proba(X_last)[:, 1][0])

    signal_df = pd.DataFrame([{
        "date":         last_row.index[0],
        "prob_16w":     round(prob_8w, 4),
        "prob_12w":     round(secondary_probs.get("target_12w", np.nan), 4),
        "prob_26w":     round(secondary_probs.get("target_26w", np.nan), 4),
        "train_rows":   len(df_train),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }])

    return model, signal_df


# ===========================================================================
# REPORT
# ===========================================================================

def print_wf_report(metrics_df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("  WALK-FORWARD — Metriche per fold (target_16w)")
    print("=" * 72)
    print(f"  {'Anno':<6} {'TrainRows':>9} {'TestRows':>8} {'%Pos':>5} {'AUC':>7} "
          f"{'LogLoss':>8} {'Prec':>6} {'Recall':>7} {'F1':>6}")
    print("  " + "-" * 69)

    auc_vals = []
    for _, row in metrics_df.iterrows():
        auc_str = f"{row['auc']:.4f}" if not pd.isna(row["auc"]) else "  n/a "
        auc_vals.append(row["auc"])
        print(
            f"  {str(row['fold']):<6}"
            f" {row['train_rows']:>9}"
            f" {row['n_test']:>8}"
            f" {row['pct_positive']:>4.1f}%"
            f" {auc_str:>7}"
            f" {row['logloss']:>8.4f}"
            f" {row['precision']:>6.3f}"
            f" {row['recall']:>7.3f}"
            f" {row['f1']:>6.3f}"
        )

    valid_aucs = [a for a in auc_vals if not np.isnan(a)]
    if valid_aucs:
        print("  " + "-" * 69)
        print(f"  {'MEAN':<6} {'':>9} {'':>8} {'':>5}"
              f" {np.mean(valid_aucs):>7.4f}"
              f" {metrics_df['logloss'].mean():>8.4f}"
              f" {metrics_df['precision'].mean():>6.3f}"
              f" {metrics_df['recall'].mean():>7.3f}"
              f" {metrics_df['f1'].mean():>6.3f}")
        print(f"  {'STD':<6} {'':>9} {'':>8} {'':>5}"
              f" {np.std(valid_aucs):>7.4f}")

    print(f"\n  Top 15 feature per importanza media (Gain LightGBM):")
    print(f"  {'#':<4} {'FEATURE':<45} {'IMP_MEAN':>9}  {'IMP_STD':>8}")
    print("  " + "-" * 72)
    for i, row in importance_df.head(15).iterrows():
        print(f"  {i+1:<4} {row['feature']:<45} {row['importance_mean']:>9.2f}  {row['importance_std']:>8.2f}")

    print()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 72)
    print("  MODELLO ENSEMBLE — LightGBM Walk-Forward")
    print(f"  Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Caricamento dati
    print("\n  Caricamento dataset e feature list...")
    df, features = load_data()
    print(f"  Dataset: {df.shape[0]} righe × {df.shape[1]} colonne")
    print(f"  Feature selezionate: {len(features)}")

    # Controllo target
    target_col = TARGET_PRIMARY
    n_valid = df[target_col].notna().sum()
    print(f"  Righe con {target_col} valido: {n_valid}")

    # ------------------------------------------------------------------ #
    # Walk-forward
    # ------------------------------------------------------------------ #
    print(f"\n  Avvio walk-forward (expanding window)...")
    predictions_df, metrics_df, importance_df = run_walk_forward(df, features, target_col)

    if predictions_df.empty or metrics_df.empty:
        print("  ERRORE: nessun fold completato.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Modello corrente
    # ------------------------------------------------------------------ #
    print(f"\n  Addestramento modello corrente (tutti i dati disponibili)...")
    current_model, signal_df = train_current_model(df, features, target_col)
    print(f"  Segnale corrente:")
    print(f"    Data:      {signal_df['date'].values[0]}")
    print(f"    prob_16w:   {signal_df['prob_16w'].values[0]:.4f}")
    print(f"    prob_12w:   {signal_df['prob_12w'].values[0]:.4f}")
    print(f"    prob_26w:   {signal_df['prob_26w'].values[0]:.4f}")

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print_wf_report(metrics_df, importance_df)

    # ------------------------------------------------------------------ #
    # Salvataggio
    # ------------------------------------------------------------------ #
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pred_path   = RESULTS_DIR / "wf_predictions.csv"
    metrics_path = RESULTS_DIR / "wf_metrics.csv"
    imp_path    = RESULTS_DIR / "feature_importance.csv"
    signal_path = RESULTS_DIR / "current_signal.csv"
    model_path  = MODELS_DIR  / "lgbm_current.pkl"

    predictions_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(imp_path, index=False)
    signal_df.to_csv(signal_path, index=False)

    with open(model_path, "wb") as f:
        pickle.dump(current_model, f)

    print(f"  ✓ Salvato: {pred_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Salvato: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Salvato: {imp_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Salvato: {signal_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Salvato: {model_path.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
