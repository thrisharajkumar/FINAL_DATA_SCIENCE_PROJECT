# Exploratory Data Analysis (EDA)

This folder contains exploratory notebooks used to understand the MOSFET simulation data, extracted datasheet parameters, and EMI output targets before detailed preprocessing and model training. The aim of this stage is to gain insights into the input/output feature distributions, detect anomalies, and confirm consistency across devices.

---

## 1. EDA_Features_Understanding.ipynb

**Objective**  
- Perform exploratory data analysis on simulation inputs and EMI targets.  
- Understand statistical properties, correlations, and anomalies.  

**Methods**  
- Visualize feature distributions (histograms, density plots, boxplots).  
- Correlation heatmaps between simulation inputs and EMI targets.  
- Identify anomalies such as negative rise times or extreme outliers.  
- Compare input feature ranges across devices.  

**Outputs**  
- Visual understanding of input and target behavior.  
- Initial list of anomalies and inconsistencies for later cleaning.  
- Feature–target correlations to guide feature engineering.  

---

## 2. input_summary_per_mosfet.ipynb

**Objective**  
- Summarize and compare simulation input parameters across MOSFET devices.  
- Confirm shared simulation setups across devices for balanced dataset merging.  

**Methods**  
- Compute descriptive statistics (mean, min, max, std) per MOSFET.  
- Plot input parameter ranges (e.g., Vbus, Rg, Ls4–Ls11).  
- Compare distributions of inputs across all six MOSFETs.  
- Detect inconsistencies where devices do not overlap in simulation setups.  

**Outputs**  
- Statistical summary tables of inputs per MOSFET.  
- Visual confirmation of common simulation spaces.  
- Basis for defining balanced and comparable datasets for merging.  

---

## Summary

Together, these EDA notebooks:  
1. Provide an initial understanding of simulation inputs and EMI outputs.  
2. Identify anomalies and outliers for later cleaning.  
3. Confirm which simulation setups are shared across MOSFETs to allow consistent dataset merging.  

These insights directly support the preprocessing and feature engineering pipelines.
