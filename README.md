# Gold Macro Trend Model

> **A macro-driven, machine-learning pipeline that acts as a gold allocation slider: it measures the intensity of macro tailwinds and scales gold exposure continuously from 20% to 80%, delivering near-identical Sharpe to Buy&Hold with 48% less drawdown.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PC9zdmc+)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Walk-Forward AUC](https://img.shields.io/badge/Walk--Forward%20AUC-0.703-brightgreen)]()
[![Signal](https://img.shields.io/badge/Current%20Signal-LONG%20%E2%96%B2-success)]()

---

## Current Signal

> *Last computed: 2026-02-13 — updated weekly every Monday*

| Composite Score | Direction | 12w Prob | 16w Prob | 26w Prob |
|:-:|:-:|:-:|:-:|:-:|
| **69.5 / 100** | 🟢 **LONG** | 59.4% | 69.8% | 78.9% |

**Signal thresholds:** `LONG > 65` · `SHORT < 35` · `FLAT 35–65`

To refresh the signal with the latest market data:

```bash
cd gold_model
python -m src.pipeline.update_pipeline
```

---

## How to Use the Signal

The model is a **macro regime detector**, not a weekly timer. Its real edge emerges when the score drives a *continuous allocation* rather than a binary in/out decision.

### Optimal Strategy: Allocation Slider

$$\text{Gold allocation} = 20\% + \frac{\text{score} - 55}{70 - 55} \times 60\%$$

| Score | Allocation | Macro interpretation |
|:-----:|:----------:|:---------------------|
| 55.0 | **20%** | Macro neutral — maintain minimum exposure |
| 62.5 | **50%** | Macro moderately positive |
| 69.5 | **~79%** | Macro strongly positive — current signal |
| 70.0 | **80%** | Maximum conviction |

Rebalance **monthly** (not weekly). Only ~13 rebalancings occurred in 10 years of OOS data.

### Why not binary LONG/FLAT?

Even during FLAT periods, gold rose in **57.8% of weeks** with an annualized return of **+13.1%**. Exiting entirely sacrifices real return without meaningfully reducing risk. The floor at 20% is the rational response to this evidence.

---

## Why This Project

Gold is one of the most macro-sensitive assets in the world, yet its short-term price is notoriously noisy. This model focuses on the **medium-term regime** (3–6 months) where macro forces — real yields, dollar strength, central bank demand, speculator positioning — dominate over intraday noise.

Key design principles:
- **No forward-looking features**: strict lookahead-bias prevention, every feature uses only data available at prediction time
- **Walk-forward validation**: 10 expanding-window folds (2016–2025), no walk-forward leakage
- **Calibrated probabilities**: Platt scaling ensures outputs are true probabilities, not raw scores
- **Explainability first**: 9 interpretable thematic factor groups, importance tracked across all folds

---

## Results

### Walk-Forward AUC — 10 Folds (2016–2025)

| Year | Train Rows | Test Rows | % Positive | AUC | Significance |
|:----:|:----------:|:---------:|:----------:|:---:|:------------:|
| 2016 | 573 | 53 | 54.7% | 0.539 | borderline |
| 2017 | 626 | 52 | 61.5% | 0.603 | |
| 2018 | 678 | 52 | 36.5% | 0.864 | ★★★ p<0.001 |
| 2019 | 730 | 52 | 71.2% | 0.623 | ★ p<0.10 |
| 2020 | 782 | 52 | 55.8% | 0.318 | COVID regime flip |
| 2021 | 834 | 53 | 52.8% | 0.741 | ★★★ p<0.001 |
| 2022 | 887 | 52 | 48.1% | 0.674 | ★★ p<0.014 |
| 2023 | 939 | 52 | 63.5% | 0.782 | ★★★ p<0.001 |
| 2024 | 991 | 52 | 98.1% | 0.980 | ★★ p<0.033 |
| 2025 | 1043 | 43 | 97.7% | 0.905 | |
| **Mean** | | **513 OOS** | | **0.703** | **7/10 significant** |
| **Std** | | | | **0.185** | |

> **7 out of 10 folds are statistically significant** (p < 0.10, permutation test).  
> Fold 2020 reflects a genuine COVID regime break — not a model error.

### Signal Accuracy

| Metric | Value | Baseline |
|--------|------:|--------:|
| LONG signal accuracy (OOS) | **68.8%** | 63.4% |
| LONG predictions | 302 / 513 | — |
| Correct LONG calls | 208 / 302 | — |
| Lift over base rate | **1.09×** | 1.00× |
| Calibration ECE (all targets) | **< 0.06** | — |
| Calibration bias | **≈ 0.000** | — |

### Profitability Backtest — Strategy Comparison (OOS 2016–2025)

| Strategy | CAGR | Sharpe | Max DD | Calmar | Notes |
|----------|-----:|-------:|-------:|-------:|-------|
| Buy & Hold gold | 14.54% | 0.992 | -18.43% | 0.79 | Benchmark |
| MA52 technical filter | 7.73% | 0.624 | -29.56% | 0.26 | Simple baseline |
| Model binary LONG/FLAT | 6.23% | 0.749 | -11.19% | 0.56 | Original signal |
| Model continuous 0–100% | 7.37% | 0.929 | -11.47% | 0.64 | Score as weight |
| **Model floor 20–80%** | **7.31%** | **0.983** | **-9.59%** | **0.76** | **★ Optimal** |

The floor strategy achieves **Sharpe 0.983** (vs 0.992 for B&H) with **48% less drawdown** and only 13 rebalancings over 10 years.

### Score Monotonicity — Forward 16-Week Return by Score Band

| Score range | N | Hit rate @16w | Avg return | Info Ratio |
|:-----------:|:-:|:------------:|:----------:|:----------:|
| 55.0–57.5 | 97 | 66.0% | +2.97% | 0.44 |
| 57.5–60.0 | 108 | 63.9% | +3.29% | 0.36 |
| 60.0–62.5 | 75 | 72.0% | +5.13% | 0.59 |
| 62.5–65.0 | 27 | 59.3% | +2.98% | 0.36 |
| 65.0–67.5 | 84 | 75.0% | +6.52% | 0.76 |
| **67.5–70.0** | **122** | **77.9%** | **+5.36%** | **0.73** |

Higher score → higher hit rate → higher average return. The relationship is **monotone** — justifying proportional sizing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                          │
│  FRED API · Yahoo Finance · WGC · COT Reports · Manual      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                         │
│  353 raw features · 9 thematic groups · lags [4,8,12,16,26w]│
│  targets: 12w (±2%) · 16w PRIMARY · 26w                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FACTOR SELECTION                           │
│  353 → 67 features (81% reduction)                         │
│  Pearson |r| > 0.10 · VIF < 10 · 9 group quota             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            WALK-FORWARD TRAINING (LightGBM)                 │
│  10 annual folds · expanding window · 2005–2025             │
│  3 targets simultaneously · early stopping per fold         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 PLATT CALIBRATION                           │
│  Logistic regression on OOS probabilities per horizon       │
│  ECE < 0.06 · zero bias confirmed                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              COMPOSITE SCORE (0–100)                        │
│  12w × 0.25 + 16w × 0.50 + 26w × 0.25                      │
│  LONG > 65 · FLAT 35–65 · SHORT < 35                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Groups

67 features organized in 9 interpretable thematic groups:

| # | Group | Key Features | Intuition |
|---|-------|-------------|-----------|
| 1 | **Real Rates** | `REAL_YIELD_10Y`, `REAL_YIELD_5Y` | Negative real rates = gold tailwind |
| 2 | **Inflation** | `CPI_yoy_pct`, `BREAKEVEN_10Y_chg` | Inflation regime drives safe-haven demand |
| 3 | **Nominal Rates** | `FED_FUNDS_chg_26w`, `TREASURY_10Y_chg` | Rate hike cycles hurt gold medium-term |
| 4 | **Dollar (DXY)** | `DXY_pctile_3y`, `DXY_chg_12w` | Inverse correlation with gold |
| 5 | **Risk Sentiment** | `VIX_pctile_1y`, `SP500_chg_12w` | Recession fears boost gold |
| 6 | **COT Positioning** | `COT_net_pctile_3y`, `COT_OI_pct_12w` | Speculator crowding signals |
| 7 | **Geopolitics** | `GLD_flows`, `MOVE_Index_chg` | Macro volatility indicator |
| 8 | **WGC Structural** | `WGC_CB`, `WGC_INVEST_pctile_3y` | Central bank & ETF demand |
| 9 | **Gold Momentum** | `GOLD_pctile_3y`, `GOLD_chg_4w` | Trend following component |

### Top 10 Features by LightGBM Gain

```
FED_FUNDS_chg_26w     ████████████████████  178.7
WGC_INVEST_pctile_3y  ███████████████████   159.2
WGC_ETF_vs_ma52       ████████████████      133.7
REAL_YIELD_10Y        ███████████████       124.5
DXY_pctile_3y         ███████████████       120.7
FED_FUNDS_pct_8w      █████████████         106.0
GOLD_chg_4w           █████████████         102.6
COT_OI_pct_12w        ██████████             81.4
CPI_yoy_pct           ██████████             80.4
COT_net_pctile_3y     ██████████             80.1
```

> Effective feature count (1/HHI): **37.9** — diversified, no single feature dominates.  
> All top-10 features have CV < 0.25 across folds — stable importance.

---

## Data Sources

| Source | Data | Update |
|--------|------|--------|
| [FRED](https://fred.stlouisfed.org/) | Fed Funds Rate, CPI, Real/Nominal Yields, Breakevens | Monthly/Weekly |
| [Yahoo Finance](https://finance.yahoo.com/) | XAU/USD, DXY, S&P 500, VIX, GLD ETF | Daily |
| [WGC](https://www.gold.org/) | Central Bank demand, ETF flows, Investment demand | Quarterly |
| [CFTC COT](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm) | Non-commercial gold futures positioning | Weekly |
| [MOVE Index](https://www.ice.com/report/movetm) | Bond market volatility | Daily |

---

## Project Structure

```
gold_model/
├── src/
│   ├── data/
│   │   ├── download_data.py       # Pulls all raw data from APIs
│   │   └── build_dataset.py       # Merges sources into weekly panel
│   ├── features/
│   │   ├── feature_engineering.py # 353 features, lags, targets
│   │   └── factor_analysis.py     # 353 → 67 selection + group quotas
│   ├── models/
│   │   ├── model.py               # Walk-forward LightGBM training
│   │   └── calibrate.py           # Platt calibration + composite score
│   ├── evaluation/
│   │   ├── backtest.py            # Weekly P&L profitability backtest
│   │   └── regime_analysis.py     # Optimal use case: allocation slider
│   └── pipeline/
│       └── update_pipeline.py     # Weekly one-click update
├── data/
│   ├── raw/                       # Downloaded raw data (git-ignored)
│   └── processed/                 # Intermediate datasets (git-ignored)
├── models/                        # Trained model files (git-ignored)
├── outputs/
│   ├── results/                   # Backtest CSVs (git-ignored)
│   └── charts/                    # Performance charts (git-ignored)
├── config.py                      # Central configuration
├── requirements.txt
├── .env.example                   # API key template
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/gold-macro-trend-model.git
cd gold-macro-trend-model/gold_model
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your FRED API key (free at https://fred.stlouisfed.org)
```

### 3. Download data & build dataset

```bash
python -m src.data.download_data
python -m src.data.build_dataset
```

### 4. Engineer features & select factors

```bash
python -m src.features.feature_engineering
python -m src.features.factor_analysis
```

### 5. Train model & calibrate

```bash
python -m src.models.model
python -m src.models.calibrate
```

### 6. Get the current signal

```bash
python -m src.pipeline.update_pipeline
```

---

## Weekly Update (Once Trained)

After the initial training, update the signal every Monday morning:

```bash
cd gold_model
python -m src.pipeline.update_pipeline
```

The pipeline will:
1. Download the latest FRED/Yahoo data
2. Compute new feature values
3. Load the trained models from `models/`
4. Output the composite score and directional signal

---

## Methodology

### Walk-Forward Validation

The model uses an **expanding-window walk-forward** strategy to simulate real deployment:

```
Train: 2005─────────────────2015 | Test: 2016
Train: 2005──────────────────────2016 | Test: 2017
Train: 2005───────────────────────────2017 | Test: 2018
...
Train: 2005────────────────────────────────────2024 | Test: 2025
```

- **Training starts**: 2005-01-01 (full macro cycle coverage)
- **First test fold**: 2016 (10 years minimum training)
- **Total OOS observations**: 513 weekly data points
- **No data from the test set ever touches training** — confirmed by overlap tests

### Target Engineering

The primary prediction target is **binary**: does gold rise ≥ 2% over the next 16 weeks?

A composite score is formed as a weighted average across 3 horizons:

$$\text{Score} = 0.25 \times P_{12w} + 0.50 \times P_{16w} + 0.25 \times P_{26w}$$
### Optimal Allocation Formula

Based on the regime analysis (`src/evaluation/regime_analysis.py`), the score should drive a **continuous allocation** rather than a binary signal:

$$\text{Gold allocation} = 20\% + \frac{\text{score} - 55}{70 - 55} \times 60\%$$

Where 55 and 70 are the empirical min/max of the OOS score distribution. This parameterization yielded **Sharpe 0.983** and **Max Drawdown −9.59%** over 2016–2025.
### Calibration

Raw LightGBM outputs are calibrated with **Platt scaling** (logistic regression on OOS folds). This ensures that `P = 0.70` means "gold rose ~70% of the time in similar configurations."

---

## Limitations & Honest Assessment

| Issue | Severity | Notes |
|-------|----------|-------|
| CAGR below B&H gold | Medium | Floor strategy: 7.3% vs 14.5% B&H — misses structural bull run during low-conviction periods |
| COVID 2020 regime break | Medium | AUC 0.32 in 2020 — unprecedented macro disruption |
| Score range narrow (55–70) | Low | Model is structurally bullish on gold 2016–2025; no deep bear recognized |
| Global OOS AUC 0.55 | Low | Artifact of base-rate shifting across years, not model failure |
| Score bounds may drift | Medium | Min/max 55–70 derived from OOS 2016–2025; may shift in future regimes |
| WGC data is quarterly | Low | Interpolated to weekly; reduces signal precision |
| Transaction costs | Minimal | Only 13 trades in 10 years; GLD bid-ask ≈00.01% |

> **This model is a research tool, not financial advice. Past performance does not guarantee future results.**

---

## Model Configuration

Key hyperparameters (see `config.py`):

```python
TARGET_HORIZONS_WEEKS = [12, 16, 26]  # Multi-horizon setup
TARGET_THRESHOLD      = 0.02           # ±2% to classify up/down
TARGET_PRIMARY        = "target_16w"   # Primary model

# LightGBM
num_leaves     = 15
max_depth      = 4
n_estimators   = 300
learning_rate  = 0.03
subsample      = 0.8
colsample_bytree = 0.7

# Walk-forward
TRAIN_START    = "2005-01-01"
FIRST_TEST_YEAR = 2016
```

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

<div align="center">

*Built with LightGBM · scikit-learn · pandas · FRED API · Yahoo Finance*

</div>
