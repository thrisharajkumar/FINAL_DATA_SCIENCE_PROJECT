# EMI Surrogate Modeling with SiC MOSFETs

This repository contains the complete pipeline for building surrogate machine learning models to predict **Electromagnetic Interference (EMI)** metrics from **SiC MOSFET datasheet parameters** and **LTSpice simulation data**.  

The project forms the basis of a dissertation that explores **generalization across multiple MOSFET devices**, using both **classical machine learning** and **deep learning models**.

## Motivation

Wide-bandgap (WBG) devices such as **Silicon Carbide (SiC)** and **Gallium Nitride (GaN) MOSFETs** enable compact, efficient, and high-performance power converters for electric vehicles. However, their rapid switching transitions introduce steep **dv/dt** and **di/dt**, which interact with parasitic inductances and capacitances to cause **overshoot, undershoot, ringing, and electromagnetic interference (EMI)**.  

Traditional EMI evaluation relies on **double-pulse test (DPT) simulations** in LTSpice with MATLAB post-processing. While accurate, this approach is **computationally heavy and slow**, especially when scaling to multiple devices, parameter sweeps, and operating conditions. This creates a bottleneck for **fast design iteration** and delays EMI-aware optimisation.  

**Surrogate machine learning models** offer a scalable alternative by directly learning the nonlinear mapping between device parameters and EMI targets from simulation data. Once trained, these models provide **near real-time EMI predictions**, generalise to **unseen devices**, and support systematic exploration of **best- and worst-case switching scenarios**. Artificial Neural Networks (ANNs) in particular are well-suited for capturing the **high-dimensional nonlinear behaviours** of switching transitions.

---

## Research Aims, Novelty, and Contributions

- **Artificial Neural Networks (ANNs) for Surrogate EMI Prediction**  
  - Developed a multi-output ANN capable of predicting 13 switching waveform characteristics (rise/fall times, overshoot, undershoot, ringing).  
  - Achieved strong generalisation performance on unseen MOSFET devices.  

- **Benchmarking with Classical Baselines**  
  - Implemented Ridge, SVR, Random Forest, LightGBM, and XGBoost for multi-output regression, highlighting limitations in device generalisation.  

- **Iterative ANN Architectural Development**  
  - Progressed from Baseline MLPs → Deep MLPs → Multi-Head → Masked → Attention-based ANNs.  
  - Incorporated dropout, L2 regularisation, batch normalisation, early stopping, learning-rate scheduling, physics-informed features, and per-target scaling.  

- **Classification of Switching Behaviour**  
  - Extended surrogate modelling to identify **best/neutral/worst-case operating conditions**, supporting EMI-aware design.  

- **Harmonic Spectrum Analysis and THD**  
  - Computed harmonic spectra and **Total Harmonic Distortion (THD)** as indicators of EMI severity, linking waveform modelling to spectral EMI metrics.  

---

## Objectives

1. Extract and preprocess simulation and datasheet parameters for six SiC MOSFETs.  
2. Perform exploratory data analysis (EDA) to understand input features and EMI target characteristics.  
3. Build a **clean, balanced dataset** with consistent simulation setups across devices.  
4. Implement **feature engineering** with physics-informed derived features.  
5. Train and evaluate classical ML models (Ridge, SVR, Random Forest, LightGBM, XGBoost).  
6. Train and optimize deep learning models (ANN, attention-based, multi-headed architectures).  
7. Evaluate **generalization performance** on unseen MOSFET devices.  
8. Deploy the best surrogate model for EMI prediction and scenario analysis.

---

## Repository Structure

Please check the project_structure.txt
tree /f > project_structure.txt
---

# PHASE 1: SWITCHING WAVEFORM WAVEFORM SURROGRATE MODELLING

## Workflow Overview

1. **EDA**  
   - `EDA_Features_Understanding.ipynb`  
   - `input_summary_per_mosfet.ipynb`  

2. **Data Preprocessing**  
   - Extraction:  
     - `extract_sic_mosfet_parameters.ipynb`  
     - `extract_tables_from_datasheets_pdf.ipynb`  
   - Merging & Cleaning:  
     - `Merging_Simulation_and_MOSFET_data.ipynb`  
     - `Outlier_Negatives_and_Null_values.ipynb`  
     - `Outlier_Removal_Each_MOSFET/`  
     - `Merged_Final_Complete.ipynb`  
   - Feature Engineering:  
     - `Feature_engineering.ipynb`  
     - `feature_engineering.py`  

3. **Classical Models**  
   - Ridge Regression → `Ridge_Regression.ipynb`  
   - Support Vector Regression (SVR) → `SVR.ipynb`  
   - Random Forest → `RandomForest.ipynb`  
   - LightGBM → `LightGBM.ipynb`  
   - XGBoost → `XGBoost.ipynb`  

    Balanced FAST sampling for quick iteration
    - 70/15/15 split (via 70/30 then halved)
    - Scale outputs ONLY (inputs left in physical units)
    - Metrics (R2, RMSE, MAE) per target + overall
    - Plots: Val vs Test R² bars, Test scatter, Residual histograms

4. **Neural Network Models - ANN**

- Multiple ANN architectures developed for EMI surrogate modeling:
  - **Baseline ANN** (`BaselineANN.ipynb`) – stepwise improvements (regularization, batch norm, dropout, schedulers).
  - **Attention ANN** (`Attention_ANN.ipynb`) – integrates attention layers to focus on key input features.
  - **Deep MLP** (`Deep_MLP.ipynb`) – deeper multilayer perceptron to capture complex nonlinearities.
  - **Masked ANN** (`Masked_ANN.ipynb`) – applies feature masking per output target.
  - **Multi-Head ANN** (`Multi_head_ANN.ipynb`) – independent branches per EMI output for multi-task learning.

- **Common outputs across models**:
  - Trained `.h5` models
  - `r2_rmse_tables/` (train, val, test, unseen metrics)
  - `predicted_vs_actual/` scatter plots
  - Training/validation loss curves

These ANN models extend the classical baselines, focusing on **non-linear feature learning**, **generalization to unseen MOSFETs**, and **architectural experimentation** to maximize EMI prediction accuracy.

5. **Final Model: Multi-Head ANN + Light Attention**

**Folder:** `FINAL_MODEL/final_multihead_attention/`  
**Notebook:** `FINAL_MODEL/FINAL_MODEL.ipynb`  

- **Description:** Final surrogate model predicting **13 EMI waveform parameters**.  
- **Key Features:** Multi-head ANN, light attention, physics-informed inputs, generalisation to unseen MOSFETs.  
- **Outputs:**  
  - `models/final_multihead_attention.h5` – trained Keras model  
  - `train_val_loss_curves/loss.png` – training vs validation loss  
  - `r2_rmse_tables/` – R² & RMSE tables (`train.csv`, `val.csv`, `test_seen.csv`, `unseen.csv`)  
  - `predicted_vs_actual/test_seen_scatter.png` – scatter plots  
  - `predicted_vs_actual/test_seen_inputs_true_pred.csv` – inputs + true vs predicted  
  - `artifacts/` – scalers & column configs for inference  


6.**Switching Scenarios: Physics-Informed Severity Classification**

**Folder:** `Switching_Scenarios_Physics/`  
**Notebook:** `Switching_Scenarios_Physics/Classifier.ipynb`  
**Docs:** `Switching_Scenarios_Physics/README_Switching_Severity_Classification.md`  

- **Description:** Classifies switching scenarios as **Best**, **Neutral**, or **Worst** using **physics-informed scoring** (rise/fall times, overshoot/undershoot, ringing frequency).  
- **Per MOSFET Outputs:**  
  - `{device}_labeled.csv` – original data + z-scores + severity score + scenario label  
  - `{device}_top5_best.csv` / `{device}_top5_worst.csv` – top-5 switching cases  
  - `{device}_score_hist.png` – score distribution with thresholds  
  - `{device}_PCA_best_worst.png` – PCA scatter of switching behaviour  
  - `{device}_contribution_bar.png` – parameter contributions (Best vs Worst)  
  - `{device}_contrib_rows.csv` – per-row contribution breakdown  

7. **Full Data Run: Multi-Head ANN**

**Folder:** `Full_Data_Run_MultiHead_ANN/`  
**Notebook:** `Full_Data_Run_MultiHead_ANN/COMPLETE_RUN.ipynb`  

- **Description:** Final **complete run** of the multi-head ANN on the **entire dataset**, saving full evaluation tables, loss curves, predicted vs actual plots, and trained models.


---

## Outputs

- **Clean balanced datasets**  
  - `All_6_MOSFETs.csv` → Complete merged dataset  
  - `Train_5_MOSFETs.csv` → Training set (5 devices)  
  - `Test_1_MOSFET.csv` → Held-out unseen MOSFET  

- **EDA reports**  
  - Feature distributions  
  - Simulation input summaries per device  
  - Feature–target correlations  
---

# PHASE 2: HARMONIC SPECTRUM ANALYSIS AND TOTAL HARMONIC DISTORTION (THD)

- Module: `harmonics_pipeline.py` (see `README_Harmonics.md`)
- Input: `Harmonics_Data/harmonics_*.csv` with `Frequency_k_Hz`, `Mag_k`, `Phase_k` per row
- Steps: IQR outlier removal → THD% → dB spectrum → noise floor → decay slope → band powers (100 kHz–3 MHz, 3–10 MHz, 10–30 MHz) → peak count/max above floor → risk flags
- Outputs:
  - Per-file features: `Harmonics_Data/processed/<file>_features.csv`
  - Master table: `Harmonics_Data/processed/emi_master_features.csv`
  - IQR summary: `Harmonics_Data/processed/iqr_summary.csv`
- Plots (notebooks): harmonic components, composite waveform, spectrum bars


## Notes

- All preprocessing ensures consistent simulation setups across six MOSFET devices.  
- Final evaluation prioritizes **generalization to unseen devices**, which is critical for practical EMI modeling.  
- The repository is modular: EDA, preprocessing, feature engineering, classical ML, and ANN pipelines are separated for clarity.  

---

## References

- SiC MOSFET datasheets (Wolfspeed)  
- LTSpice simulation datasets  
- EMI and power electronics literature  