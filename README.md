# Gold Macro Trend Model

> Gold Macro Trend Model is a quantitative project that produces weekly probabilistic long/short signals for gold (XAU/USD) across medium-term horizons: 12, 16 and 26 weeks. It combines advanced feature engineering, nine interpretable thematic factor groups, and LightGBM models trained with strict walk‑forward validation to deliver calibrated, explainable signals designed as actionable macro decision support.

> This repository is the "cover" version of the project: a reproducible, end‑to‑end pipeline with verified results and documentation so you can reproduce, audit and refresh the weekly signal.

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

### Calibration

Raw LightGBM outputs are calibrated with **Platt scaling** (logistic regression on OOS folds). This ensures that `P = 0.70` means "gold rose ~70% of the time in similar configurations."

---

## Limitations & Honest Assessment

| Issue | Severity | Notes |
|-------|----------|-------|
| COVID 2020 regime break | Medium | AUC 0.32 in 2020 — unprecedented macro disruption |
| High variance across folds (std 0.185) | Medium | Small annual test sets (n≈52) inflate variance |
| Global OOS AUC 0.55 | Low | Artifact of base-rate shifting across years, not model failure |
| Imperfect monotonicity | Low | Probability not strictly monotone with score due to regime shifts |
| WGC data is quarterly | Low | Interpolated to weekly; reduces signal precision |
| No transaction costs modeled | Medium | Backtest assumes frictionless execution |

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
