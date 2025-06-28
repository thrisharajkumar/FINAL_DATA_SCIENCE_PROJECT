# Machine Learning-Based EMI Prediction in EV Power Converters

## Abstract

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

###  Project Deliverables:
- A validated ML model for EMI prediction
- A reusable dataset of simulated and datasheet parameters
- Comparative analysis of ML algorithms
- A fast, scalable framework for EMI-aware EV converter design

##  Problem Statement

Power electronic converters are vital components in electric vehicles (EVs), enabling efficient transformation of power across subsystems such as SMPS, AC-DC, DC-DC converters, and MPPT units. With rising demands for compact, energy-efficient, and high-performance systems, **wide-bandgap (WBG) semiconductors** like **SiC** and **GaN** are replacing traditional silicon devices.

These WBG devices offer fast switching and reduced power losses but generate **high-frequency EMI** due to rapid di/dt and dv/dt transitions, ringing effects, and unintended **parasitics** from PCB layout and component properties. This EMI degrades system performance and complicates converter design, especially at early stages.

Conventional EMI prediction methods rely heavily on **LTSpice simulations** and **Double Pulse Testing (DPT)**, which are time-intensive and not feasible for iterative or scalable hardware development.

---
##  Project Aims and Checklist

| Aim                                                                 | Checklist Item                                                                 |
|----------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Predict EMI characteristics (overshoot, ringing, dv/dt, di/dt)      | [ ] Define EMI-related outputs as target variables                            |
| Build dataset using simulations and datasheets                      | [ ] Run LTSpice Double Pulse Tests (DPT)                                      |
|                                                                      | [ ] Extract key device parameters (Qg, Vth, RDS(on), Coss)                    |
| Perform feature engineering                                          | [ ] Derive features from PCB layout and gate drive parameters                 |
| Train and compare ML models                                          | [ ] Train Support Vector Machines (SVM)                                       |
|                                                                      | [ ] Train Artificial Neural Networks (ANN)                                    |
| Evaluate model performance                                           | [ ] Compute RMSE, MAE, and R²                                                  |
|                                                                      | [ ] Generate predicted vs actual plots                                        |
|                                                                      | [ ] Analyze feature importance (SHAP/permutation)                             |
| Validate with real PCB test data                                     | [ ] Collect oscilloscope EMI measurements                                     |
|                                                                      | [ ] Compare ML predictions vs hardware measurements                           |
| Deliver reusable outputs                                             | [ ] Prepare clean dataset for future use                                      |
|                                                                      | [ ] Document and share ML tool + evaluation results                           |
--- 

