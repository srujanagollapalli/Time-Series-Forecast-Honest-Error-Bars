
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Time-Series Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time-Series Forecast with Honest Error Bars")
st.write(
    "Explore daily temperature data, baseline forecasts, "
    "prediction intervals, and forecasting errors."
)

uploaded_file = st.file_uploader(
    "Upload DailyDelhiClimateTrain.csv",
    type=["csv"]
)

data_path = Path("DailyDelhiClimateTrain.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
elif data_path.exists():
    df = pd.read_csv(data_path, parse_dates=["date"])
else:
    st.info("Upload DailyDelhiClimateTrain.csv to begin.")
    st.stop()

required = {"date", "meantemp"}
if not required.issubset(df.columns):
    st.error("CSV must contain 'date' and 'meantemp' columns.")
    st.stop()

df = df[["date", "meantemp"]].dropna()
df = df.sort_values("date").reset_index(drop=True)

if len(df) < 10:
    st.error("Please upload a dataset with at least 10 valid rows.")
    st.stop()

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

forecast_value = train["meantemp"].iloc[-1]
predictions = np.full(len(test), forecast_value)
errors = test["meantemp"].to_numpy() - predictions

mae = float(np.mean(np.abs(errors)))
rmse = float(np.sqrt(np.mean(errors ** 2)))

st.subheader("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Training Records", f"{len(train):,}")
c3.metric("Test Records", f"{len(test):,}")

st.subheader("Historical Mean Temperature")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["date"], df["meantemp"], label="Actual Temperature")
ax.axvline(test["date"].iloc[0], linestyle="--", label="Test Period")
ax.set_xlabel("Date")
ax.set_ylabel("Mean Temperature (°C)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.subheader("Baseline Forecast Evaluation")
c1, c2 = st.columns(2)
c1.metric("Test MAE", f"{mae:.3f} °C")
c2.metric("Test RMSE", f"{rmse:.3f} °C")

results = test.copy()
results["prediction"] = predictions
results["absolute_error"] = np.abs(errors)

st.write("The baseline predicts the last training temperature for every test date.")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(results["date"], results["meantemp"], label="Actual")
ax2.plot(results["date"], results["prediction"], label="Baseline Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Mean Temperature (°C)")
ax2.legend()
ax2.grid(alpha=0.3)
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Prediction Interval Experiment")
st.write(
    "Choose a target coverage level. The interval is calibrated using "
    "the last part of the training period and a constant baseline forecast."
)

coverage = st.slider("Target interval coverage (%)", 80, 99, 90)
cal_split = int(len(train) * 0.8)
calibration = train.iloc[cal_split:]
calibration_fit = train.iloc[:cal_split]

if calibration_fit.empty or calibration.empty:
    st.error("Not enough training data for interval calibration.")
    st.stop()

cal_pred = calibration_fit["meantemp"].iloc[-1]
cal_errors = np.abs(calibration["meantemp"].to_numpy() - cal_pred)
quantile = float(np.quantile(cal_errors, coverage / 100))

lower = predictions - quantile
upper = predictions + quantile
actual_coverage = np.mean(
    (test["meantemp"].to_numpy() >= lower) &
    (test["meantemp"].to_numpy() <= upper)
) * 100
width = float(np.mean(upper - lower))

c1, c2, c3 = st.columns(3)
c1.metric("Observed Test Coverage", f"{actual_coverage:.1f}%")
c2.metric("Average Interval Width", f"{width:.2f} °C")
c3.metric("Target Coverage", f"{coverage}%")

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(test["date"], test["meantemp"], label="Actual")
ax3.plot(test["date"], predictions, label="Forecast")
ax3.fill_between(
    test["date"], lower, upper,
    alpha=0.2, label="Prediction Interval"
)
ax3.set_xlabel("Date")
ax3.set_ylabel("Mean Temperature (°C)")
ax3.legend()
ax3.grid(alpha=0.3)
st.pyplot(fig3)
plt.close(fig3)

st.warning(
    "Wide intervals can achieve high coverage but may be less informative. "
    "This is a simple baseline demonstration, not a guarantee of future accuracy."
)

results["lower_bound"] = lower
results["upper_bound"] = upper

st.subheader("Forecast Results")
st.dataframe(results.tail(20), use_container_width=True)

csv = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Forecast Results CSV",
    data=csv,
    file_name="temperature_forecast_with_intervals.csv",
    mime="text/csv"
)
