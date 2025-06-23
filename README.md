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
