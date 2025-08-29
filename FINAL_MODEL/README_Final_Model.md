# Final Model: Multi-Head ANN + Light Attention (Generalisation-Focused)

This script implements the **final surrogate neural network model** for predicting **13 EMI waveform parameters** from simulation and MOSFET datasheet inputs.  
It combines **multi-head output branches** with a **light attention mechanism** and **physics-informed features** to maximize **generalization performance** to unseen MOSFET devices.

---

## Key Features

- **Multi-Head Architecture**  
  Each EMI target (13 outputs) has an independent head, enabling target-specific learning while sharing a common backbone.

- **Light Attention (SE-Gating)**  
  A squeeze-and-excitation attention block is applied on the shared representation, reweighting important input channels.

- **Physics-Informed Features**  
  Derived inputs included alongside raw simulation/datasheet features:  
  - `f_resonance` (loop LC resonance, MHz)  
  - `overshoot_est`  
  - `dVdt_est`  
  - `dIdt_est`  
  - `z_loop_est` (loop impedance proxy)  
  - `rg_effective` (effective gate-drive proxy)

- **Generalisation Strategy**  
  - **Seen devices:** 5 MOSFETs used for training/validation.  
  - **Unseen device:** 1 MOSFET (`C2M0040120D`) fully held out for testing.  
  - Ensures evaluation on a device never seen during training.

- **Target Scaling & Transformations**  
  - **Log-transform** applied to heavy-tailed targets (rise times, current rise, ringing frequency).  
  - **StandardScaler** for most outputs, **MinMaxScaler** for `ringing_frequency_MHz`.

- **Loss Weighting**  
  Higher weights for weak/difficult targets (rise/fall times, ringing frequency) to improve per-head balance.

---

## Data Flow

1. **Input CSV:**  
   `merged_train_5_MOSFETs_10percent_balanced.csv`  
   - Contains simulation, datasheet, and derived physics features.

2. **Splits:**  
   - **Train/Validation:** 70/15% split of 5 seen MOSFETs.  
   - **Test (Seen):** All 5 seen MOSFETs combined.  
   - **Test (Unseen):** Entire 6th MOSFET device.

3. **Preprocessing:**  
   - Drop identifiers (`DeviceID`, `MOSFET`, `Part_Number`).  
   - Standardize all inputs globally (fit on train + unseen).  
   - Scale outputs per target with appropriate scaler + log transform.  

---

## Training

- Optimizer: **Adam** (`lr=5e-4`)  
- Loss: **MSE per target**, with custom loss weights  
- Regularization:  
  - L2 kernel regularization  
  - Batch Normalization  
  - Dropout (0.2)  
- EarlyStopping + ReduceLROnPlateau callbacks  
- Epochs: up to 160 (with patience-based stopping)  
- Batch size: 256  

---

## Outputs

All results are saved under `final_multihead_attention/`:

- **Models**  
  - `models/final_multihead_attention.h5` (trained Keras model)

- **Training Curves**  
  - `train_val_loss_curves/loss.png`

- **Evaluation Tables** (`r2_rmse_tables/`)  
  - `train.csv` – metrics on training split  
  - `val.csv` – metrics on validation split  
  - `test_seen.csv` – metrics on seen MOSFETs (internal test)  
  - `unseen.csv` – metrics on unseen MOSFET

- **Predicted vs Actual** (`predicted_vs_actual/`)  
  - `test_seen_scatter.png` – scatter plots for all 13 targets  
  - `test_seen_inputs_true_pred.csv` – full table with inputs + true + predicted outputs

- **Artifacts (for inference)** (`artifacts/`)  
  - `input_scaler.pkl`  
  - `output_scalers.pkl` (per-target scalers + log flags)  
  - `INPUT_COLUMNS.json`, `PHYS_COLS.json`, `TARGET_COLUMNS.json`

---

## Purpose

This model is the **final dissertation submission model**:  

- Achieves **robust accuracy across all 13 EMI targets**.  
- Prioritises **generalisation to unseen MOSFETs**, not just memorisation of seen devices.  
- Encodes **physics knowledge + learned attention** for improved interpretability and predictive stability.  
- Produces a **reusable artifact set** (scalers, model, columns) for inference and deployment.  
