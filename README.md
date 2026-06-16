# 📈 Sales Quantity Forecasting with ARIMA

> Time series analysis and demand forecasting on retail transaction data using ARIMA modeling

![Python](https://img.shields.io/badge/Python-3.7+-blue?style=flat&logo=python)
![pandas](https://img.shields.io/badge/pandas-1.x-150458?style=flat&logo=pandas)
![statsmodels](https://img.shields.io/badge/statsmodels-ARIMA-green?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-metrics-orange?style=flat&logo=scikit-learn)

---

## 📌 Overview

This project applies **ARIMA (AutoRegressive Integrated Moving Average)** to forecast monthly product demand from retail invoice data. It covers the full pipeline: data cleaning, feature engineering, time series resampling, autocorrelation analysis, model training, and visual evaluation.

**Business value**: Accurate demand forecasting reduces stockouts, improves procurement planning, and directly impacts revenue.

---

## 🔍 What the script does

1. **Loads** raw retail transaction data from `dataset.csv`
2. **Cleans** the data — removes cancelled orders (negative quantities), zero-price records, and empty rows
3. **Engineers** a `Total_Price` column (`Quantity × UnitPrice`) and extracts year-month from invoice dates
4. **Aggregates** daily transactions into monthly quantity totals
5. **Plots** the time series and an autocorrelation chart to identify seasonality and lag patterns
6. **Trains** an ARIMA(2,2,0) model on 66% of the data and forecasts the remaining 34%
7. **Evaluates** forecast accuracy using Mean Squared Error (MSE)
8. **Visualizes** actual vs predicted quantities side by side

---

## 📁 Project structure

```
sales-arima-forecast/
├── forecast.py        # Main analysis and forecasting script
├── dataset.csv        # Input data (retail transaction records)
└── README.md
```

---

## 📊 Dataset format

The script expects a CSV file named `dataset.csv` with at least these columns:

| Column | Description | Example |
|---|---|---|
| `InvoiceDate` | Date of transaction | `2011-01-15 10:30:00` |
| `Quantity` | Number of units sold | `12` |
| `UnitPrice` | Price per unit | `3.75` |

Compatible with the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) — download it and rename to `dataset.csv`.

---

## 🚀 Getting started

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/sales-arima-forecast.git
cd sales-arima-forecast
```

**2. Install dependencies**
```bash
pip install pandas matplotlib seaborn statsmodels scikit-learn numpy
```

**3. Add your dataset**

Place your `dataset.csv` in the project root directory.

**4. Run the script**
```bash
python forecast.py
```

---

## 📤 Output

Running the script produces three sequential plots:

**Monthly quantity trend** — aggregated sales volume over time, useful for spotting growth trends and seasonality.

**Autocorrelation plot** — shows how strongly past values correlate with future ones, used to validate ARIMA lag parameters.

**Actual vs predicted** — blue line shows real test values; red line shows ARIMA forecasts. Closer lines = better model fit.

The terminal also prints step-by-step predicted vs expected values and a final MSE score:

```
predicted=45231.2, expected=47800.0
predicted=48102.6, expected=46500.0
...
Test MSE: 3241502.847
```

---

## ⚙️ Model configuration

The ARIMA model is configured as `ARIMA(p=2, d=2, q=0)`:

| Parameter | Value | Meaning |
|---|---|---|
| `p` | 2 | Uses the last 2 time steps as autoregressive inputs |
| `d` | 2 | Differences the series twice to achieve stationarity |
| `q` | 0 | No moving average component |

To tune these parameters, inspect the autocorrelation plot output — it reveals the optimal lag structure for your specific dataset.

---

## 🛠 Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and time series resampling |
| `matplotlib` | Plotting |
| `seaborn` | Color palette |
| `statsmodels` | ARIMA model |
| `scikit-learn` | MSE evaluation metric |
| `numpy` | Array operations |

---

## 🔭 Possible improvements

- Grid search over `(p, d, q)` using AIC/BIC to find optimal parameters automatically
- Add SARIMA to capture seasonal patterns (monthly/quarterly cycles)
- Compare against Prophet or LSTM baselines
- Forecast individual product categories rather than aggregate quantity
- Export predictions to CSV for BI tool integration

---

## 📄 License

MIT — free to use and adapt with attribution.
