# Classical Machine Learning Models

This folder contains implementations of **classical supervised multi-output regression models**. These serve as **baselines** for predicting switching waveform transients and EMI characteristics from MOSFET parameters and simulation data.  

Each model is trained and evaluated on the cleaned, balanced datasets using consistent splits and evaluation metrics.

---

## 1. Ridge Regression (Multi-output Baseline)

- **Notebook:** `Ridge_Regression.ipynb`  
- Data split: **70/15/15**  
- Preprocessing: standardized features  
- Outputs: metrics for train, validation, and test  

**Plots Produced**  
1. R² bar plots (Train vs Validation, light blue)  
2. Scatter plots (Test set)  
3. Residual histograms  
4. Mean |coef| (Ridge feature importance, light blue)  

---

## 2. Support Vector Regression (SVR)

- **Notebook:** `SVR.ipynb`  
- Model: **LinearSVR + MultiOutputRegressor**  
- Balanced sampling by unique `(Vbus … Ls11)` combinations  
- Data split: **70/15/15**  
- Preprocessing: standardize **X and y**  
- One SVR per target using `MultiOutputRegressor`  

**Metrics**  
- R², RMSE, MAE for train/validation/test  

**Plots Produced**  
1. R² bar plots (Validation vs Test)  
2. Test scatter plots  
3. Residual histograms  
4. Residuals vs predicted plots  
5. Mean |coef| (linear feature importance)  

---

## 3. Random Forest (Multi-output Baseline)

- **Notebook:** `RandomForest.ipynb`  
- Data split: **70/15/15**  
- Sampling: fast per-combination sampling  
- Metrics: R², RMSE, MAE  

**Plots Produced**  
1. R² bar plots (Validation vs Test: pink vs light blue)  
2. Test scatter plots (dark blue + red dashed 45° reference line)  
3. Residual histograms (Test only, light blue + dashed zero line)  
4. Mean Random Forest feature importance (light blue)  

---

## 4. LightGBM (Multi-output Baseline)

- **Notebook:** `LightGBM.ipynb`  
- Data split: **70/15/15**  
- Metrics: R², RMSE, MAE  

**Plots Produced**  
1. R² bar plots (Validation vs Test: pink vs light blue)  
2. Test scatter plots (dark blue + red dashed 45° reference line)  
3. Residual histograms (Test only, light blue + dashed zero line)  
4. Mean LightGBM feature importance averaged across outputs (light blue)  

---

## 5. XGBoost (Multi-output Baseline)

- **Notebook:** `XGBoost.ipynb`  
- Balanced FAST sampling for quick iteration  
- Data split: **70/15/15** (via 70/30 then halved)  
- Preprocessing: scale **outputs only**, inputs remain in physical units  
- Metrics: **R², RMSE, MAE** per target and overall  

**Plots Produced**  
1. Validation vs Test R² bar plots  
2. Test scatter plots  
3. Residual histograms  
4. Mean XGBoost feature importance (averaged across outputs)  
5. SHAP analysis:  
   - Mean |SHAP| bar plot  
   - Top-3 SHAP dependence plots  

---

## Evaluation Strategy

- **Splits**:  
  - Train (70%), Validation (15%), Test (15%)  
  - Additional holdout set for unseen MOSFET generalization  

- **Metrics**:  
  - R² (coefficient of determination)  
  - RMSE (root mean squared error)  
  - MAE (mean absolute error)  

- **Outputs**:  
  - Comparative plots across models  
  - Feature importance rankings (coefficients or impurity-based)  
  - SHAP analysis (XGBoost)  

---

## Purpose

These classical models serve as **benchmark baselines** before applying advanced ANN-based surrogate models. They provide:  
- Interpretability (feature importances, coefficients, SHAP values)  
- Performance references for R², RMSE, MAE  
- Insights into the feature–output relationships  

This baseline stage establishes a strong foundation for evaluating the advantages of deep learning models in EMI surrogate prediction.
