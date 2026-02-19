"""
backtest.py — Profitability Backtest del Gold Macro Trend Model
===============================================================

Risponde alla domanda fondamentale:
    "Seguendo i segnali del modello, si guadagna?"

METODOLOGIA:
    1. STRATEGIA SETTIMANALE
       - Ogni settimana il modello emette LONG / FLAT / SHORT
       - LONG  → position = +1 (compra oro)
       - FLAT  → position =  0 (liquidità, rendimento 0%)
       - SHORT → position = -1 (vende allo scoperto)
       - Il rendimento raccolto è il return settimanale del prezzo dell'oro
         della settimana successiva all'emissione del segnale (no lookahead)

    2. STRATEGIA ORIZZONTE COMPLETO (16w)
       - Per ogni segnale LONG, si registra il return cumulato nelle 16w successive
       - Per ogni segnale SHORT, si registra il return inverso nelle 16w successive
       - Misura quante volte il segnale punta nella direzione giusta sul suo orizzonte naturale

    3. CONFRONTO CON BUY-AND-HOLD ORO
       - Stessa finestra 2016–2025
       - Nessuna leva, nessun costo di transazione (backtest idealizzato)

METRICHE:
    - CAGR, Sharpe, Sortino, Max Drawdown, Calmar Ratio
    - Win Rate, Profit Factor, numero segnali per tipo
    - Return annuale vs benchmark
    - Tabella hit rate per tipo di segnale

OUTPUT:
    outputs/results/backtest_report.txt  — report testuale completo
    outputs/charts/equity_curve.png      — curva equity (se matplotlib disponibile)

Esecuzione:
    cd gold_model
    python -m src.evaluation.backtest
"""

import sys
import warnings
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR

CHARTS_DIR   = PROJECT_ROOT / "outputs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LONG_THR  = 65.0
SHORT_THR = 35.0
WEEKS_PER_YEAR = 52.0


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_signals() -> pd.DataFrame:
    """Carica le predizioni calibrate OOS (2016–2025)."""
    p = RESULTS_DIR / "calibrated_predictions.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} non trovato.\nEseguire prima: python -m src.models.calibrate"
        )
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def load_gold_prices() -> pd.Series:
    """
    Carica prezzi chiusura settimanale oro da GOLD_FUTURES.csv.
    Ritorna una Series indicizzata per data (venerdì settimanali).
    """
    raw_path = PROJECT_ROOT / "data" / "raw" / "yfinance" / "GOLD_FUTURES.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} non trovato.")

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Resample a frequenza settimanale usando il venerdì (W-FRI → chiusura settimana)
    weekly = df["close"].resample("W-FRI").last().dropna()
    return weekly


# ===========================================================================
# ALIGNMENT
# ===========================================================================

def align_signals_price(
    signals: pd.DataFrame,
    gold_weekly: pd.Series,
) -> pd.DataFrame:
    """
    Allinea segnali e prezzi settimanali.
    Il segnale emesso in settimana T cattura il rendimento T+1 (no lookahead).

    Colonne output:
        signal, score, prob_cal_16w, gold_close, gold_ret_1w, gold_fwd_16w_ret
    """
    # Usa il venerdì più vicino per ogni data segnale
    # I segnali sono lunedì/venerdì → troviamo il prezzo corrispondente
    df = signals[["signal", "score", "prob_cal_16w"]].copy()

    # Merge asincrono: per ogni data nel signals, prende il prezzo del venerdì successivo
    gold_w = gold_weekly.copy()
    gold_w.index.name = "date"

    # Reindex sui segnali usando merge_asof per trovare il prezzo del venerdì vicino
    df = df.sort_index()
    gold_w = gold_w.sort_index()

    merged = pd.merge_asof(
        df,
        gold_w.rename("gold_close").reset_index(),
        left_index=True,
        right_on="date",
        direction="nearest",
        tolerance=pd.Timedelta("6 days"),
    )
    merged = merged.set_index(merged.index)
    # Riallinea per sicurezza
    merged.index = df.index

    # Return settimanale: shift(-1) = return della prossima settimana
    merged["gold_ret_1w"] = merged["gold_close"].pct_change(1).shift(-1)

    # Return forward 16 settimane: valore effettivo nella colonna actual_16w
    if "actual_16w" in signals.columns:
        merged["actual_16w"] = signals["actual_16w"]
    if "gold_fwd_16w" in signals.columns:
        merged["gold_fwd_16w"] = signals["gold_fwd_16w"]

    # Carica anche dal dataset processato il return grezzo +16w
    try:
        ds_path = PROJECT_ROOT / "data" / "processed" / "dataset_selected.csv"
        ds = pd.read_csv(ds_path, index_col=0, parse_dates=True)
        ds.index = pd.to_datetime(ds.index).tz_localize(None)
        if "gold_fwd_16w" in ds.columns:
            merged["gold_fwd_16w_ret"] = ds["gold_fwd_16w"].reindex(merged.index)
    except Exception:
        pass

    return merged


# ===========================================================================
# STRATEGIA SETTIMANALE
# ===========================================================================

def weekly_strategy(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Strategia settimanale: position basata sul segnale della settimana precedente.
    L'avversario di riferimento è Buy & Hold.

    Return:
        DataFrame con colonne: position, strategy_ret, bnh_ret, strategy_cum, bnh_cum
    """
    df = aligned.copy()

    # Mappa segnale → posizione
    signal_map = {"LONG": 1.0, "FLAT": 0.0, "SHORT": -1.0}
    df["position"] = df["signal"].map(signal_map).fillna(0.0)

    # Shift +1: usiamo il segnale della settimana T per catturare il return T+1
    df["position_lagged"] = df["position"].shift(1).fillna(0.0)

    # Return strategia = posizione × return oro settimana dopo
    df["strategy_ret"] = df["position_lagged"] * df["gold_ret_1w"]
    df["bnh_ret"]      = df["gold_ret_1w"]

    # Rimuovi le ultime righe (NaN per gold_ret_1w)
    df = df.dropna(subset=["gold_ret_1w"])

    # Curve cumulate (equity = 1 iniziale)
    df["strategy_cum"] = (1 + df["strategy_ret"]).cumprod()
    df["bnh_cum"]      = (1 + df["bnh_ret"]).cumprod()

    return df


# ===========================================================================
# STRATEGIA ORIZZONTE 16W
# ===========================================================================

def horizon_strategy(aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni segnale direzionale (LONG/SHORT), misura quanto ha guadagnato
    tenendo il trade per 16 settimane (l'orizzonte naturale del modello).

    Restituisce DataFrame con: date, signal, score, fwd_ret_16w, hit, pnl
    """
    df = aligned[["signal", "score", "prob_cal_16w"]].copy()

    # Carica i ritorni forward 16w reali
    if "gold_fwd_16w_ret" in aligned.columns:
        df["fwd_ret_16w"] = aligned["gold_fwd_16w_ret"]
    elif "gold_fwd_16w" in aligned.columns:
        df["fwd_ret_16w"] = aligned["gold_fwd_16w"]
    else:
        # Calcolalo direttamente dai prezzi
        gold_path = PROJECT_ROOT / "data" / "raw" / "yfinance" / "GOLD_FUTURES.csv"
        g = pd.read_csv(gold_path, index_col=0, parse_dates=True)["close"]
        g.index = pd.to_datetime(g.index).tz_localize(None)
        g_w = g.resample("W-FRI").last()
        # Per ogni data nel df, prende il return 16 settimane dopo
        fwd = {}
        for dt in df.index:
            future_dt = dt + pd.Timedelta(weeks=16)
            # prezzo più vicino entro 7 giorni
            idx = g_w.index.searchsorted(future_dt)
            if idx < len(g_w):
                p0 = g_w.asof(dt)
                p1 = g_w.iloc[idx]
                if p0 and p0 > 0:
                    fwd[dt] = (p1 - p0) / p0
        df["fwd_ret_16w"] = pd.Series(fwd)

    # Solo segnali direzionali (non FLAT)
    df = df[df["signal"].isin(["LONG", "SHORT"])].copy()
    df = df.dropna(subset=["fwd_ret_16w"])

    # gold_fwd_16w è espresso in % (es. 9.24 = +9.24%) → converti in decimale
    df["fwd_ret_16w_dec"] = df["fwd_ret_16w"] / 100.0

    # P&L: LONG guadagna quando fwd > 0, SHORT guadagna quando fwd < 0
    df["pnl"] = np.where(
        df["signal"] == "LONG",
        df["fwd_ret_16w_dec"],
        -df["fwd_ret_16w_dec"],
    )
    df["hit"] = (df["pnl"] > 0).astype(int)

    return df


# ===========================================================================
# METRICHE PERFORMANCE
# ===========================================================================

def performance_metrics(returns: pd.Series, freq: float = WEEKS_PER_YEAR) -> dict:
    """Calcola metriche complete da una serie di ritorni settimanali."""
    r = returns.dropna()
    if len(r) == 0:
        return {}

    n_weeks = len(r)
    n_years = n_weeks / freq

    total_ret    = (1 + r).prod() - 1
    cagr         = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol          = r.std() * np.sqrt(freq)
    sharpe       = (r.mean() / r.std()) * np.sqrt(freq) if r.std() > 0 else np.nan

    downside     = r[r < 0].std() * np.sqrt(freq)
    sortino      = (cagr / downside) if downside > 0 else np.nan

    # Max Drawdown
    cum          = (1 + r).cumprod()
    running_max  = cum.cummax()
    drawdown     = (cum - running_max) / running_max
    max_dd       = drawdown.min()
    calmar       = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate settimanale
    # Win rate su settimane ATTIVE (posizione != 0)
    r_active     = r[r != 0]
    win_rate     = (r_active > 0).sum() / len(r_active) if len(r_active) > 0 else np.nan
    n_active     = len(r_active)
    pos_rets     = r_active[r_active > 0].sum()
    neg_rets     = abs(r_active[r_active < 0].sum())
    profit_factor = pos_rets / neg_rets if neg_rets > 0 else np.inf

    return {
        "n_weeks":       n_weeks,
        "n_active":      n_active,
        "n_years":       round(n_years, 1),
        "total_ret":     round(total_ret * 100, 2),
        "cagr":          round(cagr * 100, 2),
        "vol_ann":       round(vol * 100, 2),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "max_dd":        round(max_dd * 100, 2),
        "calmar":        round(calmar, 3),
        "win_rate":      round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
    }


def annual_returns(returns: pd.Series) -> pd.DataFrame:
    """Calcola i ritorni annuali da una serie di ritorni settimanali."""
    r = returns.dropna()
    years = r.index.year.unique()
    rows  = []
    for y in sorted(years):
        yr = r[r.index.year == y]
        rows.append({
            "year":       y,
            "n_weeks":    len(yr),
            "return_pct": round(((1 + yr).prod() - 1) * 100, 2),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# REPORT BUILDER
# ===========================================================================

def build_report(
    wf:           pd.DataFrame,  # weekly-strategy DataFrame
    hz:           pd.DataFrame,  # horizon-strategy DataFrame
    out_buf:      StringIO,
) -> None:
    """Scrive il report di profitabilità completo su out_buf."""
    p = print

    def _p(*args, **kwargs):
        kwargs["file"] = out_buf
        print(*args, **kwargs)

    def hdr(title: str, char: str = "=") -> None:
        _p("\n" + char * 65)
        _p(f"  {title}")
        _p(char * 65)

    # ── date range
    start_dt = wf.index[0].strftime("%Y-%m-%d")
    end_dt   = wf.index[-1].strftime("%Y-%m-%d")

    hdr("GOLD MACRO TREND MODEL — PROFITABILITY BACKTEST")
    _p(f"  Periodo OOS:        {start_dt}  →  {end_dt}")
    _p(f"  Settimane totali:   {len(wf)}")
    _p(f"  Strategia:          No leva | No costi transazione")
    _p(f"  Benchmark:          Buy & Hold oro")

    # ── signal distribution
    hdr("1. DISTRIBUZIONE SEGNALI", "-")
    sig_counts = wf["signal"].value_counts()
    total_sig  = len(wf)
    for s in ["LONG", "FLAT", "SHORT"]:
        n = sig_counts.get(s, 0)
        _p(f"  {s:6s}  {n:4d} settimane  ({n/total_sig*100:.1f}%)")

    # ── overall metrics
    strat_m = performance_metrics(wf["strategy_ret"])
    bnh_m   = performance_metrics(wf["bnh_ret"])

    hdr("2. PERFORMANCE GLOBALE (SETTIMANALE, 2016–2025)", "-")
    _p(f"  {'Metrica':<20} {'Strategia':>12} {'Buy&Hold':>12}")
    _p(f"  {'-'*20} {'-'*12} {'-'*12}")
    metrics_display = [
        ("Total Return %",      "total_ret",     "%"),
        ("CAGR %",              "cagr",          "%"),
        ("Volatilità ann %",    "vol_ann",        "%"),
        ("Sharpe Ratio",        "sharpe",         ""),
        ("Sortino Ratio",       "sortino",        ""),
        ("Max Drawdown %",      "max_dd",         "%"),
        ("Calmar Ratio",        "calmar",         ""),
        ("Win Rate attivo %",   "win_rate",       "%"),
        ("Profit Factor",       "profit_factor",  ""),
        ("Settimane OOS",       "n_weeks",        ""),
        ("Settimane attive",    "n_active",        ""),
    ]
    for label, key, unit in metrics_display:
        sv = strat_m.get(key, "N/A")
        bv = bnh_m.get(key, "N/A")
        sv_str = f"{sv}{unit}" if unit else str(sv)
        bv_str = f"{bv}{unit}" if unit else str(bv)
        _p(f"  {label:<20} {sv_str:>12} {bv_str:>12}")

    # ── annual returns table
    hdr("3. RITORNI ANNUALI: STRATEGIA vs BUY&HOLD", "-")
    _p(f"  {'Anno':<6} {'Segnali':>8} {'Strategia':>12} {'Buy&Hold':>12} {'Differenza':>12} {'Vincitore':>10}")
    _p(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    strat_ann = annual_returns(wf["strategy_ret"])
    bnh_ann   = annual_returns(wf["bnh_ret"])
    merged_ann = strat_ann.merge(bnh_ann, on=["year", "n_weeks"], suffixes=("_s", "_b"))

    wins_strat = 0
    wins_bnh   = 0
    for _, row in merged_ann.iterrows():
        diff = row["return_pct_s"] - row["return_pct_b"]
        winner = "Strategia" if diff > 0 else "Buy&Hold"
        if diff > 0:
            wins_strat += 1
        else:
            wins_bnh += 1
        _p(
            f"  {int(row['year']):<6} {int(row['n_weeks']):>8} "
            f"{row['return_pct_s']:>11.2f}% "
            f"{row['return_pct_b']:>11.2f}% "
            f"{diff:>+11.2f}% "
            f"{winner:>10}"
        )
    _p(f"\n  Anni vinti da Strategia: {wins_strat}/10  |  Buy&Hold: {wins_bnh}/10")

    # ── horizon hit rate
    hdr("4. HIT RATE SULL'ORIZZONTE NATURALE (16 SETTIMANE)", "-")
    if len(hz) > 0:
        overall_hit  = hz["hit"].mean() * 100
        overall_mean = hz["pnl"].mean() * 100
        overall_med  = hz["pnl"].median() * 100
        long_df      = hz[hz["signal"] == "LONG"]
        short_df     = hz[hz["signal"] == "SHORT"]

        _p(f"  Segnali direzionali totali: {len(hz)}")
        _p(f"  Hit rate globale:           {overall_hit:.1f}%  (baseline 50%)")
        _p(f"  Return medio per segnale:   {overall_mean:+.2f}%")
        _p(f"  Return mediano per segnale: {overall_med:+.2f}%")
        _p(f"  (N.B.: i trade si sovrappongono — il compounding non ha significato)")
        _p()
        if len(long_df) > 0:
            p10 = long_df["pnl"].quantile(0.10) * 100
            p90 = long_df["pnl"].quantile(0.90) * 100
            _p(f"  LONG  — n={len(long_df):3d} | hit={long_df['hit'].mean()*100:.1f}% "
               f"| avg={long_df['pnl'].mean()*100:+.2f}% "
               f"| median={long_df['pnl'].median()*100:+.2f}% "
               f"| P10={p10:+.1f}%  P90={p90:+.1f}%")
        if len(short_df) > 0:
            _p(f"  SHORT — n={len(short_df):3d} | hit={short_df['hit'].mean()*100:.1f}% "
               f"| avg={short_df['pnl'].mean()*100:+.2f}% "
               f"| median={short_df['pnl'].median()*100:+.2f}%")

        # Per score bucket
        _p()
        _p(f"  {'Score bucket':<12} {'N':>4} {'Hit rate':>10} {'Avg P&L':>10} {'Median':>10} {'P10':>8} {'P90':>8}")
        _p(f"  {'-'*12} {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        bins   = [35, 55, 65, 75, 85, 101]
        labels = ["35–55", "55–65", "65–75", "75–85", "85+"]
        hz2    = hz.copy()
        hz2["score_abs"] = hz2["score"]
        hz2["bucket"]    = pd.cut(hz2["score_abs"], bins=bins, labels=labels, right=False)
        for b in labels:
            bg = hz2[hz2["bucket"] == b]
            if len(bg) == 0:
                continue
            hit_pct = bg["hit"].mean() * 100
            avg_pnl = bg["pnl"].mean() * 100
            med_pnl = bg["pnl"].median() * 100
            p10     = bg["pnl"].quantile(0.10) * 100
            p90     = bg["pnl"].quantile(0.90) * 100
            _p(f"  {b:<12} {len(bg):>4} {hit_pct:>9.1f}% {avg_pnl:>+9.2f}% {med_pnl:>+9.2f}% {p10:>+7.1f}% {p90:>+7.1f}%")
    else:
        _p("  Dati forward 16w non disponibili per questo test.")

    # ── signal timing analysis
    hdr("5. ANALISI TIMING SEGNALI PER ANNO", "-")
    _p(f"  {'Anno':<6} {'LONG wk':>8} {'FLAT wk':>8} {'Gold ret':>10} {'In mercato':>12}")
    _p(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*12}")
    for yr in sorted(wf.index.year.unique()):
        yr_wf    = wf[wf.index.year == yr]
        n_long   = (yr_wf["signal"] == "LONG").sum()
        n_flat   = (yr_wf["signal"] == "FLAT").sum()
        gold_ann = ((1 + yr_wf["bnh_ret"]).prod() - 1) * 100
        pct_in   = n_long / len(yr_wf) * 100
        _p(f"  {yr:<6} {n_long:>8} {n_flat:>8} {gold_ann:>+9.2f}% {pct_in:>11.1f}%")

    # ── only-long strategy
    hdr("5. VARIANTE: SOLO SEGNALI LONG (NESSUNO SHORT)", "-")
    wf2 = wf.copy()
    wf2["pos_longonly"] = wf2["position"].clip(lower=0).shift(1).fillna(0)
    wf2["ret_longonly"] = wf2["pos_longonly"] * wf2["gold_ret_1w"]
    lo_m = performance_metrics(wf2["ret_longonly"])
    _p(f"  CAGR:          {lo_m.get('cagr', 'N/A'):>8}%")
    _p(f"  Sharpe:        {lo_m.get('sharpe', 'N/A'):>8}")
    _p(f"  Max DD:        {lo_m.get('max_dd', 'N/A'):>8}%")
    _p(f"  Win rate:      {lo_m.get('win_rate', 'N/A'):>8}%")
    _p(f"  Total Return:  {lo_m.get('total_ret', 'N/A'):>8}%")

    # ── drawdown analysis
    hdr("6. ANALISI DRAWDOWN STRATEGIA", "-")
    cum_s   = wf["strategy_cum"]
    peak    = cum_s.cummax()
    dd_s    = (cum_s - peak) / peak
    max_dd_date = dd_s.idxmin()
    _p(f"  Max Drawdown:   {strat_m['max_dd']:.2f}%")
    _p(f"  Data picco DD:  {max_dd_date.strftime('%Y-%m-%d')}")
    # Periodi sopra/sotto acqua
    above_water = (dd_s >= -0.01).sum()
    _p(f"  Settimane sopra acqua (DD < 1%): {above_water}/{len(dd_s)} ({above_water/len(dd_s)*100:.1f}%)")

    # ── summary verdict
    hdr("7. VERDETTO FINALE", "-")
    excess = strat_m.get("cagr", 0) - bnh_m.get("cagr", 0)
    sharpe_ok = strat_m.get("sharpe", 0) > 0.5
    hit_ok    = len(hz) > 0 and hz["hit"].mean() > 0.55
    wins_ok   = wins_strat >= 6

    _p(f"  CAGR strategia vs benchmark:  {strat_m.get('cagr','?'):>6}% vs {bnh_m.get('cagr','?'):>6}% ({excess:+.2f}% excess)")
    _p(f"  Sharpe > 0.5:  {'✓ SI' if sharpe_ok else '✗ NO'}  ({strat_m.get('sharpe','?')})")
    _p(f"  Hit rate > 55%:{' ✓ SI' if hit_ok else ' ✗ NO'}  ({len(hz)} segnali direzionali)")
    _p(f"  Anni vinti > 6/10: {'✓ SI' if wins_ok else '✗ NO'}  ({wins_strat}/10)")

    _p()
    if sharpe_ok and (hit_ok or wins_ok):
        _p("  ► VALUTAZIONE: Il modello è PROFITTEVOLE su base OOS 2016–2025.")
        _p("    Genera alpha rispetto al buy&hold con minore volatilità.")
    elif wins_ok:
        _p("  ► VALUTAZIONE: Il modello è MARGINALMENTE PROFITTEVOLE.")
        _p("    Batte il benchmark più anni che no, ma il vantaggio assoluto è limitato.")
    else:
        _p("  ► VALUTAZIONE: Il modello NON dimostra profittabilità robusta OOS.")
        _p("    Il segnale non è sufficiente per una strategia puramente meccanica.")

    _p()
    _p("  NOTE IMPORTANTI:")
    _p("  - Backtest ideale: nessun costo di transazione, nessuno slippage")
    _p("  - Short su oro richiede accesso a futures/ETF inverso → costo reale ~1-2%/anno")
    _p("  - Orizzonte modello: 16 settimane (segnale non ottimizzato per 1-week trading)")
    _p("  - Questo è un modello di REGIME, non di timing preciso")
    _p("=" * 65)


# ===========================================================================
# EQUITY CHART
# ===========================================================================

def plot_equity(wf: pd.DataFrame, out_path: Path) -> bool:
    """Salva equity curve comparison. Ritorna True se riuscito."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                                  gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle("Gold Macro Trend Model — Equity Curve OOS (2016–2025)",
                     fontsize=14, fontweight="bold", y=0.98)

        ax1, ax2, ax3 = axes

        # ── Equity curve
        ax1.plot(wf.index, wf["strategy_cum"] * 100, label="Strategia Modello",
                 color="#2196F3", linewidth=2.0)
        ax1.plot(wf.index, wf["bnh_cum"] * 100, label="Buy & Hold Oro",
                 color="#FF9800", linewidth=1.5, linestyle="--")
        ax1.set_ylabel("Equity (base 100)", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        # ── Background coloring per segnale
        signal_colors = {"LONG": "#C8E6C9", "FLAT": "#FFF9C4", "SHORT": "#FFCDD2"}
        prev_sig = None
        prev_dt  = None
        for dt, row in wf.iterrows():
            sg = row["signal"]
            if sg != prev_sig:
                if prev_sig and prev_dt:
                    ax1.axvspan(prev_dt, dt,
                                alpha=0.25,
                                color=signal_colors.get(prev_sig, "white"))
                prev_sig = sg
                prev_dt  = dt
        # Ultimo segmento
        if prev_sig and prev_dt:
            ax1.axvspan(prev_dt, wf.index[-1],
                        alpha=0.25,
                        color=signal_colors.get(prev_sig, "white"))

        # ── Score composito
        ax2.plot(wf.index, wf["score"], color="#9C27B0", linewidth=1.2)
        ax2.axhline(65, color="green", linestyle=":", linewidth=1, alpha=0.7, label="LONG thr (65)")
        ax2.axhline(35, color="red",   linestyle=":", linewidth=1, alpha=0.7, label="SHORT thr (35)")
        ax2.fill_between(wf.index, 65, 100, alpha=0.08, color="green")
        ax2.fill_between(wf.index, 0,  35, alpha=0.08, color="red")
        ax2.set_ylabel("Score (0–100)", fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(True, alpha=0.3)

        # ── Drawdown
        dd = (wf["strategy_cum"] - wf["strategy_cum"].cummax()) / wf["strategy_cum"].cummax() * 100
        ax3.fill_between(wf.index, dd, 0, color="#F44336", alpha=0.5)
        ax3.plot(wf.index, dd, color="#F44336", linewidth=0.8)
        ax3.set_ylabel("Drawdown %", fontsize=11)
        ax3.set_xlabel("Data", fontsize=11)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  [WARN] Chart non generato: {e}")
        return False


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print("  GOLD MODEL — BACKTEST PROFITTABILITÀ")
    print("=" * 65)

    # Carica dati
    print("\n[1/5] Caricamento segnali calibrati OOS...")
    signals = load_signals()
    print(f"      {len(signals)} settimane OOS ({signals.index[0].date()} → {signals.index[-1].date()})")

    print("[2/5] Caricamento prezzi oro settimanali...")
    gold_w = load_gold_prices()
    print(f"      {len(gold_w)} settimane disponibili "
          f"({gold_w.index[0].date()} → {gold_w.index[-1].date()})")

    print("[3/5] Allineamento segnali e prezzi...")
    aligned = align_signals_price(signals, gold_w)

    print("[4/5] Costruzione strategie...")
    wf = weekly_strategy(aligned)
    hz = horizon_strategy(aligned)
    print(f"      Settimane disponibili per backtest settimanale: {len(wf)}")
    print(f"      Segnali direzionali con return 16w disponibile: {len(hz)}")

    # Report
    print("[5/5] Generazione report...\n")
    buf = StringIO()
    build_report(wf, hz, buf)
    report_text = buf.getvalue()

    # Stampa sul terminale
    print(report_text)

    # Salva su file
    report_path = RESULTS_DIR / "backtest_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  Report salvato → {report_path}")

    # Equity chart
    chart_path = CHARTS_DIR / "equity_curve.png"
    ok = plot_equity(wf, chart_path)
    if ok:
        print(f"  Chart salvato   → {chart_path}")
    else:
        print("  Chart non generato (matplotlib non disponibile o errore)")

    # Salva anche i dati grezzi del backtest per analisi ulteriori
    wf.to_csv(RESULTS_DIR / "backtest_weekly.csv")
    hz.to_csv(RESULTS_DIR / "backtest_horizon.csv")
    print(f"  Dati grezzi salvati → outputs/results/backtest_weekly.csv")
    print(f"                      → outputs/results/backtest_horizon.csv")


if __name__ == "__main__":
    main()
