# Forecasting International Visitors to Australia Using SARIMA Models (1985–2005)

## Overview

This project analyzes monthly international visitor arrivals to Australia from 1985 to 2005 using Seasonal Autoregressive Integrated Moving Average (SARIMA) models.

The objective is to identify the underlying trend and seasonal patterns in the data and develop a forecasting model capable of producing accurate future predictions.

## Forecast

![Forecast](figures/Forecast_SARIMA_model.png)

## Dataset

The dataset is provided in:

```text
data/australia_visitors_1985_2005.txt
```

It contains monthly observations of international visitor arrivals to Australia between 1985 and 2005.

## Methodology

The analysis follows the standard Box–Jenkins approach:

1. Exploratory Data Analysis (EDA)

   * Visualization of the time series
   * Identification of trend and seasonality

2. Data Transformation

   * Logarithmic transformation for variance stabilization
   * First-order differencing
   * Seasonal differencing (period = 12)

3. Model Selection

   * Examination of ACF and PACF plots
   * Estimation of several SARIMA models
   * Comparison using AIC and BIC

4. Diagnostic Checking

   * Residual analysis
   * Ljung–Box tests
   * Normality assessment using QQ-plots

5. Forecasting

   * Rolling one-step-ahead evaluation
   * Generation of future forecasts with prediction intervals

## Selected Model

The final model selected is:

SARIMA(0,1,1)(1,1,1)[12]

This specification provided the best balance between model fit, forecasting performance, and parsimony.

## Main Results

* Strong upward trend detected.
* Clear annual seasonality with period 12.
* Log transformation successfully stabilized variance.
* Residual diagnostics indicated no significant autocorrelation.
* Forecasts preserve the historical seasonal pattern.
* Mean Squared Prediction Error (MSPE): approximately 0.0034.

## Repository Structure

```text
.
├── data/
│   └── 15.txt
├── analysis/
│   └── time_series_analysis.qmd
├── figures/
│   ├── ACF_PACF_plots.png
│   ├── Forecast_SARIMA_model.png
│   ├── Stationary_time_series.png
│   └── Time_series_plot.png
├── report/
│   └── time_series_report.pdf
└── README.md
```

## Requirements

R packages used:

```r
forecast
tseries
ggplot2
TSA
```

Install them with:

```r
install.packages(c(
  "forecast",
  "tseries",
  "ggplot2",
  "TSA"
))
```

## Running the Analysis

Render the Quarto document:

```bash
quarto render analysis/time_series_analysis.qmd
```

## Full Report

The complete report is available in:

- [time_series_analysis_report.pdf](report/time_series_report.pdf)

## Author

Mateus Auza Cruz
