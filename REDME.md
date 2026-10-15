
# Time-Series Forecast with Honest Error Bars

## Project Overview
This project analyzes daily climate data from Delhi, India, and evaluates time-series forecasting performance using historical mean-temperature observations.

## Objective
To forecast future temperatures, measure prediction errors, and communicate uncertainty using prediction intervals.

## Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Evaluation Results

| Metric | Result |
|---|---:|
| Test MAE | 8.001 °C |
| Test RMSE | 8.892 °C |
| Target Interval Coverage | 90% |
| Observed Test Coverage | 100% |
| Average Interval Width | 35.208 °C |

## Key Findings
The baseline forecasting model achieved a test MAE of 8.001°C and RMSE of 8.892°C. The approximate 90% prediction interval covered 100% of the test observations, but its average width was 35.208°C.

This indicates that the interval may be overly wide and less informative for precise forecasting. Actual coverage alone is not sufficient to judge interval quality; both coverage and width must be considered.

## Conclusion
The project demonstrates time-series forecasting, chronological evaluation, error measurement, and uncertainty communication. Future improvements should evaluate seasonal models and calibrate prediction intervals using time-ordered validation data.

