# 📈 Stock Prediction — Time Series Analysis

> Retail sales quantity forecasting using walk-forward ARIMA modeling with stationarity testing, autocorrelation analysis, and multi-metric evaluation

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?style=flat&logo=pandas)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-green?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat&logo=scikit-learn)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8+-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

## 📌 Overview

This project builds a **monthly sales quantity forecasting pipeline** on retail transaction data using ARIMA (AutoRegressive Integrated Moving Average). It covers every step from raw invoice records to a walk-forward forecast with full visual and statistical evaluation.

**Business value**: Accurate demand forecasting reduces overstock costs, prevents stockouts, and gives procurement teams a data-driven basis for monthly purchasing decisions.

---

## ✨ What's new in this version

| Area | Improvement |
|---|---|
| **API** | Migrated from deprecated `statsmodels.tsa.arima_model` to modern `statsmodels.tsa.arima.model` |
| **Date parsing** | Replaced removed `pandas.datetime` with `pd.to_datetime()` |
| **Metrics** | Added RMSE, MAE, and MAPE alongside MSE for richer evaluation |
| **EDA** | Added ADF stationarity test to validate differencing assumptions |
| **Charts** | Two-panel forecast plot: full series overview + test-period zoom with error shading |
| **Structure** | Refactored into 6 named functions with a `main()` entry point |
| **Robustness** | Column validation on load, `unicode_escape` encoding for retail datasets |

---

## 📁 Repository structure

```
stock-prediction-time-series-analysis/
├── main.py                  # Full forecasting pipeline
├── dataset.csv              # Primary transaction dataset
├── retail_dataset.xlsx      # Raw retail data (Excel format)
├── autocorrelation.png      # Autocorrelation plot output
├── data_per_month.png       # Monthly quantity trend chart
├── prediction.png           # Actual vs predicted forecast chart
└── README.md
```

---

## 🔍 Pipeline steps

```
Load CSV  →  Clean & validate  →  Monthly aggregation
    →  EDA + ADF test  →  Walk-forward ARIMA  →  Evaluate + Plot
```

| Step | Function | What it does |
|---|---|---|
| Load | `load_data()` | Reads CSV, validates required columns exist |
| Clean | `clean_data()` | Drops returns, zero prices, blanks; creates `Total_Price` |
| Aggregate | `build_monthly_series()` | Groups invoices into monthly quantity totals, resamples to Month-Start |
| EDA | `run_eda()` | Plots trend + autocorrelation, runs ADF stationarity test |
| Forecast | `forecast_arima()` | Walk-forward ARIMA — trains on expanding window, predicts one step ahead |
| Visualise | `plot_results()` | Two-panel chart saved as `prediction.png` |

---

## 📊 Output charts

### Monthly sales quantity trend + autocorrelation
Aggregated monthly totals with autocorrelation lags — used to confirm stationarity and select ARIMA parameters.

![Monthly Data](data_per_month.png)

### Actual vs predicted — full view and test zoom
Top panel shows the full series with train/test split marked. Bottom panel zooms into the test period with error shading and metrics in the title.

![Prediction](prediction.png)

---

## 📋 Dataset format

The script expects a CSV file named `dataset.csv` with at least these columns:

| Column | Type | Description |
|---|---|---|
| `InvoiceDate` | string / datetime | Transaction date |
| `Quantity` | integer | Units sold per line item |
| `UnitPrice` | float | Price per unit |

Compatible with the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) — download and rename to `dataset.csv`.

---

## 🚀 Getting started

**1. Clone the repository**
```bash
git clone https://github.com/Benmwaura/stock-prediction-time-series-analysis.git
cd stock-prediction-time-series-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn openpyxl
```

**3. Run the pipeline**
```bash
python main.py
```

The terminal prints step-by-step predictions and a metrics summary. Two charts are saved automatically and displayed.

---

## ⚙️ ARIMA model configuration

Configured as `ARIMA(p=2, d=2, q=0)` — editable at the top of `main.py`:

```python
ARIMA_ORDER  = (2, 2, 0)   # (p, d, q)
TRAIN_SPLIT  = 0.66         # 66% train / 34% test
RESAMPLE_FREQ = "MS"        # Month-Start resampling
```

| Parameter | Value | Meaning |
|---|---|---|
| `p` | 2 | Autoregressive order — uses last 2 months as input |
| `d` | 2 | Differencing order — applied twice to remove trend |
| `q` | 0 | No moving average component |

The autocorrelation plot output helps guide parameter tuning. The spike visible around lag 10–11 suggests potential seasonal structure worth exploring with SARIMA.

---

## 📐 Evaluation metrics

The model is evaluated on the held-out test set using four metrics:

| Metric | Description |
|---|---|
| **MSE** | Mean Squared Error — penalises large errors heavily |
| **RMSE** | Root MSE — same unit as the target (quantities), easier to interpret |
| **MAE** | Mean Absolute Error — average absolute deviation per month |
| **MAPE** | Mean Absolute Percentage Error — scale-independent, good for business reporting |

Sample terminal output:
```
┌─────────────────────────────┐
│  Test MSE  :    3,241,502   │
│  Test RMSE :        1,800   │
│  Test MAE  :        1,420   │
│  Test MAPE :         8.3%   │
└─────────────────────────────┘
```

---

## 🛠 Dependencies

```
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
statsmodels>=0.14
scikit-learn>=1.4
openpyxl>=3.1
```

---

## 🔭 Future improvements

- **Parameter tuning** — grid search over `(p, d, q)` using AIC/BIC scoring
- **SARIMA** — capture seasonal monthly/quarterly cycles visible in autocorrelation
- **Prophet** — Facebook's forecasting library as a no-differencing benchmark
- **Per-product forecasting** — model individual SKUs rather than aggregate quantity
- **Streamlit dashboard** — interactive forecast explorer with date range selector
- **CI/CD pipeline** — scheduled daily retraining on fresh transaction data

---

## 👤 Author

**Benmwaura**
[GitHub Profile](https://github.com/Benmwaura)

---

## 📄 License

MIT — free to use and adapt with attribution.
