# Data Preprocessing and EDA

This folder contains all exploratory and preprocessing steps required to construct a clean, balanced dataset for EMI surrogate modeling from raw MOSFET datasheet parameters and LTSpice simulation data.

---

## Pipeline Overview

1. **EDA and Feature Understanding**  
   - Initial exploratory data analysis to understand simulation inputs, MOSFET parameters, and EMI target distributions.  

2. **Data Extraction**  
   - Extract parameters from MOSFET datasheets and parse simulation inputs.  

3. **Data Merging**  
   - Merge simulation results with datasheet parameters.  
   - Clean negative, null, and inconsistent values.  
   - Perform outlier analysis and removal for each MOSFET.  
   - Merge all MOSFET datasets into a single balanced dataset.  

4. **Feature Engineering**  
   - Create physics-informed derived features and prepare structured inputs for machine learning models.  

---

## 1. EDA and Feature Understanding

- **`EDA_Features_Understanding.ipynb`**  
  Exploratory analysis of simulation inputs and EMI targets, feature distributions, correlations, and anomalies.  

- **`input_summary_per_mosfet.ipynb`**  
  Summarizes simulation input parameter ranges per MOSFET and confirms common simulation setups.  

For detailed EDA documentation, see [README_EDA.md](README_EDA.md).  

---

## 2. Data Extraction

- **`extract_sic_mosfet_parameters.ipynb`**  
  Extracts SiC MOSFET parameters from datasheets and structures them into tabular format.  

- **`extract_tables_from_datasheets_pdf.ipynb`**  
  Parses datasheets (PDF format) to extract structured parameter tables automatically.  

---

## 3. Data Merging

- **`Merging_Simulation_and_MOSFET_data.ipynb`**  
  Combines simulation results with the extracted MOSFET datasheet parameters into a unified dataset.  

- **`Outlier_Negatives_and_Null_values.ipynb`**  
  Cleans the merged dataset by removing negative values, dropping nulls, and resolving inconsistencies.  

- **`Outlier_Removal_Each_MOSFET/`**  
  Contains per-device outlier analysis notebooks. Each focuses on one MOSFET, analyzing distributions, detecting extreme values, and removing error-prone data points.  

- **`Merged_Final_Complete.ipynb`**  
  Final merging and balancing of datasets across all MOSFETs.  
  - Ensures common simulation setups across devices.  
  - Balanced dataset rows per device:  
    - C2M0025120D → 86,335 rows  
    - C2M0040120D → 86,335 rows  
    - C2M0080120D → 86,335 rows  
    - C2M0160120D → 86,335 rows  
    - C2M0280120D → 86,335 rows  
    - C2M1000170D → 86,335 rows  
  - Outputs:  
    - `All_6_MOSFETs.csv` → Complete merged dataset  
    - `Train_5_MOSFETs.csv` → Training set (5 devices)  
    - `Test_1_MOSFET.csv` → Held-out unseen test device (C2M1000170D)  

---

## 4. Feature Engineering

- **`Feature_engineering.ipynb`**  
  Notebook version of feature engineering experiments and derivations (e.g., physics-informed features).  

- **`feature_engineering.py`**  
  Script version for a reproducible feature engineering pipeline.  

---

This pipeline ensures a clean, consistent, and balanced dataset across six MOSFETs, preparing it for ANN training, generalization testing, and EMI surrogate modeling.
