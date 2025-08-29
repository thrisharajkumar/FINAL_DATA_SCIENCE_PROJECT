# Neural Network Models

This folder contains multiple **ANN architectures** developed and tested for EMI surrogate modeling.  
Each model type explores different design choices (baseline, deep, attention, masked, multi-head) with iterative improvements.  
Every experiment includes saved models, evaluation tables (R², RMSE), predicted vs actual plots, and training/validation loss curves.

---

## 1. Baseline ANN

- **Notebook:** `BaselineANN.ipynb`  
- **Description:** Initial feedforward ANN baseline with incremental iterations.  
- **Iterations:**  
  - **First Iteration:** `baseline_ann.h5` – simple ANN without advanced regularization.  
  - **Second Iteration:** `no_regularization_ann.h5` – baseline with no regularization.  
  - **Third Iteration:** `bn_l2_ann.h5` – batch normalization + L2 regularization.  
  - **Fourth Iteration:** `dropout_l2_scheduler_ann.h5` – dropout, L2 regularization, learning rate scheduler.  
  - **Fifth Iteration (Final):** `iteration5_final_ann.h5` – final tuned baseline ANN.  

**Outputs per iteration:**  
- `models/` – saved ANN weights (.h5).  
- `predicted_vs_actual/` – scatter plots (internal test).  
- `r2_rmse_tables/` – metrics CSVs (train, val, test, unseen).  
- `train_val_loss_curves/` – training/validation loss curve.  

---

## 2. Attention ANN

- **Notebook:** `Attention_ANN.ipynb`  
- **Description:** Incorporates attention layers to improve focus on critical input features.  
- **Iterations:**  
  - **Iteration 1:** `attention_ann.h5` – first attempt at attention mechanism.  
  - **Iteration 2:** `attention_ann.h5` – refined attention ANN.  

**Outputs:**  
- Saved models, R²/RMSE tables, and loss curves for each iteration.  

---

## 3. Deep MLP

- **Notebook:** `Deep_MLP.ipynb`  
- **Description:** Deeper feedforward ANN with multiple hidden layers to capture complex patterns.  
- **Iterations:**  
  - **Iteration 1:** `iteration1_Deep_mlp.h5`  
  - **Iteration 2:** `iteration2_Deep_MLP.h5`  

**Outputs:**  
- `models/` – saved ANN weights.  
- `predicted_vs_actual/` – scatter plots for test set.  
- `r2_rmse_tables/` – metrics CSVs (train, val, test, unseen).  
- `train_val_loss_curves/` – training/validation loss curves.  

---

## 4. Masked ANN

- **Notebook:** `Masked_ANN.ipynb`  
- **Description:** Implements **feature masking** per output target based on input–output relevance mapping.  
- **Iteration:**  
  - `masked_ann.h5` – single iteration model.  

**Outputs:**  
- Metrics (R²/RMSE) across train, val, test, unseen sets.  
- Training/validation loss curves.  

---

## 5. Multi-Head ANN

- **Notebook:** `Multi_head_ANN.ipynb`  
- **Description:** Multi-headed architecture with independent output branches, designed for multi-output regression.  
- **Iterations:**  
  - **Iteration 2:** `multihead_ann.h5`  
  - **Iteration 3:** `multihead_ann.h5` (further improvements).  
  - **Fast Mode:** `multihead_ann.h5` trained on reduced dataset for quick testing.  

**Outputs:**  
- `models/` – saved ANN weights.  
- `predicted_vs_actual/` – scatter plots.  
- `r2_rmse_tables/` – metrics CSVs.  
- `train_val_loss_curves/` – training/validation loss curves.  

---

## Common Outputs Across All Models

For each iteration of each ANN type, the following are consistently saved:  

- **`models/`** → Trained ANN (`.h5`)  
- **`r2_rmse_tables/`** → CSVs with R² and RMSE (train/val/test/unseen)  
- **`predicted_vs_actual/`** → Scatter plots comparing predictions vs ground truth (where applicable)  
- **`train_val_loss_curves/`** → Training/validation loss curves (`loss.png`)  

---

## Purpose

These neural network models extend beyond the **classical baselines** by:  
- Capturing **non-linear dependencies** between MOSFET inputs and EMI targets.  
- Improving **generalization to unseen MOSFET devices**.  
- Exploring **architectural variations** (attention, deep, masked, multi-head) to maximize predictive accuracy.  

Each iteration documents how design choices (regularization, embeddings, masking, attention) affect performance, providing a progression from baseline ANN to advanced architectures.
