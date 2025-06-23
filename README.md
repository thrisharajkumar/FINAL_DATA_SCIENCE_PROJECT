# 🚗 Machine Learning-Based EMI Prediction in EV Power Converters

## 📄 Abstract

This project focuses on the growing challenge of **Electromagnetic Interference (EMI)** prediction in electric vehicle (EV) power electronic converters using **wide-bandgap (WBG)** semiconductor devices like **Silicon Carbide (SiC)** and **Gallium Nitride (GaN)**.

While WBG devices enable compact, high-efficiency designs through fast switching, they also introduce high-frequency EMI due to ringing and parasitic effects. Traditional EMI estimation via **Double Pulse Test (DPT)** simulations is time-consuming and not scalable for early-stage design iterations.

To address this, the project proposes a **machine learning-based surrogate modeling** approach using:

- **Support Vector Machines (SVM)**
- **Artificial Neural Networks (ANN)**
- (Optionally) **CNNs** and **transfer learning** for waveform-based learning

The model is trained on:
- LTSpice simulation outputs
- Semiconductor datasheet parameters (Si, SiC, GaN)

**Key features** include gate resistance, parasitic inductance, overshoot, and ringing frequency.

### ✅ Project Deliverables:
- A validated ML model for EMI prediction
- A reusable dataset of simulated and datasheet parameters
- Comparative analysis of ML algorithms
- A fast, scalable framework for EMI-aware EV converter design

## ⚙️ Problem Statement

Power electronic converters are vital components in electric vehicles (EVs), enabling efficient transformation of power across subsystems such as SMPS, AC-DC, DC-DC converters, and MPPT units. With rising demands for compact, energy-efficient, and high-performance systems, **wide-bandgap (WBG) semiconductors** like **SiC** and **GaN** are replacing traditional silicon devices.

These WBG devices offer fast switching and reduced power losses but generate **high-frequency EMI** due to rapid di/dt and dv/dt transitions, ringing effects, and unintended **parasitics** from PCB layout and component properties. This EMI degrades system performance and complicates converter design, especially at early stages.

Conventional EMI prediction methods rely heavily on **LTSpice simulations** and **Double Pulse Testing (DPT)**, which are time-intensive and not feasible for iterative or scalable hardware development.

---

## 🔍 Motivation

The need for a **faster, scalable, and data-driven EMI prediction method** is critical to accelerating the design of next-generation EV converters. Manually modeling or simulating each design permutation is inefficient, particularly when EMI is influenced by:

- Semiconductor parameters (e.g., Qg, RDS(on), Vth, Coss)
- Gate driver circuits
- Parasitic components from PCB design
- Load and switching conditions

This project proposes to use **machine learning** as a surrogate modeling strategy. By learning from simulation data and real datasheet parameters, ML models can predict EMI-related metrics such as **overshoot voltage**, **ringing frequency**, **dv/dt**, and **di/dt**—significantly speeding up early-stage design decisions.

The overall motivation is to improve **design speed**, **accuracy**, and **sustainability** of EV power electronics through predictive intelligence.

## 🎯 Aims and Objectives

### 🎯 Project Aim

To develop a **machine learning-based surrogate model** capable of predicting **Electromagnetic Interference (EMI)** characteristics—such as overshoot voltage, ringing frequency, dv/dt, and di/dt—in **Electric Vehicle (EV) onboard chargers**, using data from simulations and semiconductor datasheets.

---

### ✅ Key Objectives

- 📊 **Simulate switching behavior** using LTSpice-based Double Pulse Test (DPT) circuits with parametric variations.
- 📄 **Extract component-level parameters** (e.g., Qg, Vth, RDS(on), Coss) from datasheets of Si, SiC, and GaN MOSFETs.
- 🧮 **Engineer features** representing gate drive, PCB layout, and parasitic elements from the testbench configuration.
- 🤖 **Train and evaluate ML models** including:
  - Support Vector Machines (SVM)
  - Artificial Neural Networks (ANN)
  - (Optional) CNNs and transfer learning for waveform/image-based learning
- 🧪 **Validate model performance** against real oscilloscope measurements from hardware tests.

The ultimate goal is to build a robust, scalable, and interpretable ML pipeline that can accelerate EMI prediction and optimize the design cycle for EV power converters.
## 🧠 Approach and Dataset Design

This project builds a dataset that captures the relationship between **semiconductor-level inputs** and **EMI-related outputs** using both simulations and datasheet-driven parameters.

---

### 🧾 Data Sources

- **LTSpice Simulations**  
  Double Pulse Test (DPT) circuits are used to simulate switching behaviors with varied parameters:
  - Gate resistance (Rg)
  - Bus voltage (Vbus)
  - Parasitic inductance (Ls)
  - Load types

- **Semiconductor Datasheets**  
  Parameters are extracted for devices based on **Silicon (Si)**, **Silicon Carbide (SiC)**, and **Gallium Nitride (GaN)**, including:
  - Gate charge (Qg)
  - Threshold voltage (Vth)
  - On-resistance (RDS(on))
  - Output/input capacitance (Coss)

---

### 🎯 Prediction Targets (Labels)

The machine learning models will predict the following EMI-relevant outputs:
- ⚡ Peak overshoot voltage
- 🔁 Ringing frequency
- ⏫ Rise/fall time
- 📈 Switching slopes (dv/dt and di/dt)

This approach combines **simulation-based waveform metrics** with **datasheet parameters** to build a rich, feature-engineered dataset suitable for ML-based EMI prediction.

## 🧪 Model Strategy and Evaluation

This section outlines the modeling workflow, starting with interpretable baselines and progressing to more complex neural models. Model performance is quantitatively evaluated and compared.

---

### 🔧 Initial Modeling Steps

- **Linear Regression** and **Support Vector Regression (SVR)** as simple and interpretable baselines.
- **Feedforward Artificial Neural Networks (ANN)** using:
  - `MLPRegressor` (scikit-learn) or
  - Custom architectures in PyTorch

Key techniques include:
- Layer tuning (number of neurons and depth)
- Dropout for regularization
- Early stopping to prevent overfitting

---

### 📊 Evaluation Metrics

Models will be evaluated using:
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Predicted vs Actual Scatter Plots**
- **Feature importance** using SHAP or permutation analysis

---

### 🔍 Advanced Model Exploration

If ANN performance is suboptimal:
- Try other regressors like:
  - Random Forest Regressor
  - Gradient Boosting
  - k-Nearest Neighbours (k-NN)
- Explore **1D Convolutional Neural Networks (CNNs)** on waveform segments
- Apply **transfer learning** with pretrained models (e.g., ResNet) if EMI waveforms are stored as images

---

### 🧪 Ground Truth Validation

Final models will be validated against **real oscilloscope measurements** from PCB hardware to assess generalisation and real-world reliability.

## 🧾 Expected Contributions

This project is designed to deliver a meaningful and reusable toolkit for EMI prediction in EV power converter design.

---

### 🛠️ Key Deliverables

- ✅ A **Python-based machine learning tool** for EMI prediction using simulation and datasheet parameters.
- 📂 A **clean, well-structured dataset** combining:
  - Time-domain waveform features
  - Component-level specifications (Si, SiC, GaN)
- 📊 A **comparative study** of multiple ML models, including:
  - Linear Regression
  - Support Vector Machines (SVM)
  - Artificial Neural Networks (ANN)
  - (Optionally) CNNs and Transfer Learning methods
- 📈 A **validation report** with:
  - Predicted vs actual EMI comparisons
  - Visualizations and plots
  - Accuracy metrics (RMSE, MAE, R²)

---

This work contributes to the domain of **hardware-aware EV converter design**, offering a scalable solution for fast EMI prediction, early design optimization, and sustainable development cycles.
## 🗓️ Project Timeline

A structured 13-week plan guides the end-to-end development of the EMI prediction tool, from literature review to model validation and report writing.

---

### 📚 Weeks 1–2: Literature Review
- Understand EMI in WBG converters (SiC/GaN)
- Study switching behavior, ringing, parasitics, gate drive
- Identify key semiconductor parameters (Qg, RDS(on), Coss, Vgs, etc.)
- Research ML techniques applicable to EMI spectrum prediction

📤 **Output:** Conceptual grounding and key term definitions

---

### 🗂️ Weeks 3–4: Data Collection & Feature Engineering
- Run LTSpice simulations for Si, SiC, and GaN MOSFETs
- Extract datasheet parameters
- Merge simulation and datasheet data
- Perform initial feature engineering

📤 **Output:** Combined dataset in Excel with raw and derived features

---

### 🔧 Week 5: Final Dataset & Baseline Code
- Finalise cleaned, scaled dataset
- Review initial project setup
- Build baseline SVM model

📤 **Output:** Ready-to-use ML dataset and project review

---

### 🤖 Weeks 6–7: Baseline ML Models
- Train and evaluate Linear Regression and SVM
- Metrics: MAE, RMSE, R²
- Compare baseline models

📤 **Output:** Short evaluation report with comparison plots and scores

---

### 🧠 Weeks 8–9: ANN Development
- Build ANN model using MLPRegressor or PyTorch
- Tune hyperparameters (layers, activation, dropout)
- Compare ANN with baseline models

📤 **Output:** Performance plots and training metrics

---

### 🔬 Weeks 10–11: Extended Models & Real Data Validation
- Test additional models: Random Forest, Gradient Boosting, k-NN
- Explore: PCA, SHAP, 1D CNNs, and transfer learning
- Validate against real oscilloscope readings from PCB tests

📤 **Output:** Full comparison table and benchmarked EMI predictions

---

### 📝 Weeks 12–13: Final Report & Submission
- Compile all results, graphs, and visualisations
- Finalise modeling and clean up noisy data
- Supervisor review, proofreading, formatting

📤 **Output:** Complete MSc thesis and code package for submission

