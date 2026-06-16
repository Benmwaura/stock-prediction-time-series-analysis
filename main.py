"""
Stock / Retail Sales Forecasting — Time Series Analysis
========================================================
Pipeline: Data loading → Cleaning → EDA → ARIMA forecasting → Evaluation
Author : Benmwaura
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.arima.model import ARIMA          # ← modern statsmodels API
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH   = "dataset.csv"
ARIMA_ORDER    = (2, 2, 0)          # (p, d, q) — tune via autocorrelation plot
TRAIN_SPLIT    = 0.66               # 66% train / 34% test
RESAMPLE_FREQ  = "MS"               # Month-Start resampling
PLOT_STYLE     = "seaborn-v0_8-whitegrid"
FIGURE_SIZE    = (12, 5)

plt.style.use(PLOT_STYLE)
sns.set_palette("tab10")


# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the retail dataset and validate required columns."""
    print(f"[1/5] Loading data from '{path}' ...")
    df = pd.read_csv(path, encoding="unicode_escape", low_memory=False)

    required = {"InvoiceDate", "Quantity", "UnitPrice"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"      ✅ {len(df):,} rows loaded  |  columns: {list(df.columns)}")
    return df


# ─── 2. DATA CLEANING ─────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancellations, bad prices, and blank rows."""
    print("[2/5] Cleaning data ...")
    before = len(df)

    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    df = df[df["Quantity"]  > 0]   # drop returns / cancellations
    df = df[df["UnitPrice"] > 0]   # drop zero-price entries

    # Drop rows where ALL string columns are empty
    str_cols = df.select_dtypes(include="object").columns
    df = df[~(df[str_cols] == "").all(axis=1)]

    df["Total_Price"] = df["Quantity"] * df["UnitPrice"]

    after = len(df)
    print(f"      ✅ {before - after:,} rows removed  |  {after:,} rows retained")
    return df


# ─── 3. FEATURE ENGINEERING & AGGREGATION ────────────────────────────────────

def build_monthly_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate daily transactions into a monthly quantity time series."""
    print("[3/5] Building monthly time series ...")

    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True)
    df["YearMonth"]   = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("YearMonth")["Quantity"]
          .sum()
          .resample(RESAMPLE_FREQ)
          .mean()
          .bfill()          # back-fill any gaps
    )
    monthly.name = "Quantity"

    print(f"      ✅ {len(monthly)} monthly periods  |  "
          f"{monthly.index[0].date()} → {monthly.index[-1].date()}")
    return monthly


# ─── 4. EXPLORATORY DATA ANALYSIS ─────────────────────────────────────────────

def run_eda(series: pd.Series) -> None:
    """Plot time series trend, autocorrelation, and run ADF stationarity test."""
    print("[4/5] Running EDA ...")

    # ── Descriptive stats ────────────────────────────────────────
    print(f"\n      Descriptive statistics:\n{series.describe().round(2).to_string()}\n")

    # ── ADF stationarity test ─────────────────────────────────────
    adf_result = adfuller(series.dropna())
    print(f"      ADF Statistic : {adf_result[0]:.4f}")
    print(f"      p-value       : {adf_result[1]:.4f}")
    verdict = "✅ Stationary" if adf_result[1] < 0.05 else "⚠️  Non-stationary (differencing needed)"
    print(f"      Verdict       : {verdict}\n")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle("Exploratory Data Analysis", fontsize=14, fontweight="bold")

    # Monthly trend
    axes[0].plot(series.index, series.values, color="#378ADD", linewidth=2)
    axes[0].fill_between(series.index, series.values, alpha=0.12, color="#378ADD")
    axes[0].set_title("Monthly Sales Quantity Trend")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Quantity Sold")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=30)

    # Autocorrelation
    autocorrelation_plot(series, ax=axes[1], color="#E24B4A")
    axes[1].set_title("Autocorrelation Plot")
    axes[1].axhline(0,  color="black", linewidth=0.8)
    axes[1].axhline(0.75,  color="gray",  linestyle="--", linewidth=0.8)
    axes[1].axhline(-0.75, color="gray",  linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig("data_per_month.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("      📁 Saved: data_per_month.png")


# ─── 5. ARIMA FORECASTING ─────────────────────────────────────────────────────

def forecast_arima(series: pd.Series) -> dict:
    """
    Walk-forward ARIMA forecasting.
    Trains on expanding window; predicts one step ahead at each test point.
    """
    print(f"[5/5] Training ARIMA{ARIMA_ORDER} model ...")

    values = series.values
    split  = int(len(values) * TRAIN_SPLIT)
    train, test = values[:split], values[split:]

    history     = list(train)
    predictions = []

    for t in range(len(test)):
        model     = ARIMA(history, order=ARIMA_ORDER)
        model_fit = model.fit()
        forecast  = model_fit.forecast(steps=1)
        yhat      = float(forecast[0])

        predictions.append(yhat)
        history.append(test[t])

        print(f"      step {t+1:>2}/{len(test)}  |  "
              f"predicted: {yhat:>10,.1f}  |  actual: {test[t]:>10,.1f}")

    predictions = np.array(predictions)
    test_dates  = series.index[split:]

    # ── Metrics ───────────────────────────────────────────────────
    mse  = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(test, predictions)
    mape = np.mean(np.abs((test - predictions) / np.where(test == 0, 1, test))) * 100

    print(f"\n      ┌─────────────────────────────┐")
    print(f"      │  Test MSE  : {mse:>14,.2f}  │")
    print(f"      │  Test RMSE : {rmse:>14,.2f}  │")
    print(f"      │  Test MAE  : {mae:>14,.2f}  │")
    print(f"      │  Test MAPE : {mape:>13,.2f}%  │")
    print(f"      └─────────────────────────────┘")

    return {
        "test":        test,
        "predictions": predictions,
        "dates":       test_dates,
        "metrics":     {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape},
        "full_series": series,
        "split":       split,
    }


# ─── 6. RESULTS VISUALISATION ─────────────────────────────────────────────────

def plot_results(results: dict) -> None:
    """Plot full series with train/test split and overlaid forecast."""
    series = results["full_series"]
    split  = results["split"]
    dates  = results["dates"]
    test   = results["test"]
    preds  = results["predictions"]
    m      = results["metrics"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("ARIMA Sales Forecast — Actual vs Predicted", fontsize=14, fontweight="bold")

    # ── Top panel: full series + forecast ───────────────────────
    axes[0].plot(series.index[:split], series.values[:split],
                 color="#378ADD", linewidth=2, label="Training data")
    axes[0].plot(dates, test,
                 color="#1D9E75", linewidth=2, label="Actual (test)")
    axes[0].plot(dates, preds,
                 color="#E24B4A", linewidth=2, linestyle="--", label="ARIMA forecast")
    axes[0].axvline(series.index[split], color="gray", linestyle=":", linewidth=1.5,
                    label="Train / test split")
    axes[0].set_title("Full Series with Forecast Overlay")
    axes[0].set_ylabel("Monthly Quantity")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=30)

    # ── Bottom panel: test period zoom ──────────────────────────
    axes[1].plot(dates, test,  color="#1D9E75", linewidth=2.5,
                 marker="o", markersize=5, label="Actual")
    axes[1].plot(dates, preds, color="#E24B4A", linewidth=2.5,
                 marker="s", markersize=5, linestyle="--", label="Predicted")
    axes[1].fill_between(dates, test, preds, alpha=0.12, color="#E24B4A", label="Error band")
    axes[1].set_title(
        f"Test Period Zoom  |  RMSE: {m['RMSE']:,.1f}  |  "
        f"MAE: {m['MAE']:,.1f}  |  MAPE: {m['MAPE']:.1f}%"
    )
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Monthly Quantity")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("prediction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📁 Saved: prediction.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SALES FORECASTING — TIME SERIES ANALYSIS")
    print("=" * 55)

    df      = load_data(DATASET_PATH)
    df      = clean_data(df)
    series  = build_monthly_series(df)
    run_eda(series)
    results = forecast_arima(series)
    plot_results(results)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
