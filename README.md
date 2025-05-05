# Forecasting US Real GDP Using the FRED-QD Dataset

This project performs a forecasting experiment using the **FRED-QD** dataset, a comprehensive database of U.S. macroeconomic indicators frequently used in empirical macroeconomics and econometrics. The primary goal is to produce **1-quarter-ahead forecasts of the log-level of Real GDP** and evaluate forecast accuracy across different models.

---

## 📁 Dataset

- **File:** `DatiProject.xlsx`
- **Sheet `Data`:** Contains cleaned time series data, including the target variable `GDPC1` (Real GDP).
- **Sheet `DataForFactors`:** Contains high-dimensional macroeconomic variables for use in factor-augmented models.

---

## 🧪 Forecasting Task

- **Target Variable:**
  - `yₜ = log(GDPC1ₜ)`
  - One-quarter-ahead forecast horizon.

- **Forecast Origins:**
  - Start: **1985:Q2**
  - End: **2018:Q3**
  - Total forecasts: **P = 134**
  
- **Forecasting Approach:**
  - All models are estimated using a **rolling window** strategy.
  - Forecast evaluation is based on **Root Mean Squared Error (RMSE)** calculated using the **level** of GDP (i.e., forecasts are exponentiated before error calculation).

---

## 📊 Models Implemented

1. **Random Walk (RW)**
2. **Autoregressive Model AR(4)**
3. **Vector Autoregression VAR(4)**
4. **VAR with Optimal Lag Selection (AIC-based)**
5. **AR(4)-X:** AR model augmented with factors extracted from `DataForFactors` using PCA

---

## ⚙️ Requirements

- **MATLAB** (recommended version 2020a or later)
  - Econometrics Toolbox
  - Statistics and Machine Learning Toolbox (for PCA computation)

---

## 🔗 References

- **FRED-QD Database:**
  - https://research.stlouisfed.org/econ/mccracken/fred-databases/

- **Other tools for working with FRED-QD (not used here):**
  - [R package – fbi](https://github.com/cykbennie/fbi)
  - [Python – FredMD](https://pypi.org/project/FredMD/)
  - [Julia – FredApi.jl](https://github.com/markushhh/FredApi.jl)

---

## 📌 Notes

- All macroeconomic variables are already pre-processed (e.g., transformed to stationary series where necessary).
- Forecast results are saved for further statistical evaluation or visualization.
