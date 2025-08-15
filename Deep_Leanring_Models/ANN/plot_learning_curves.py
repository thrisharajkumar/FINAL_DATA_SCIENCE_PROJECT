import os
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import HISTORY_PATH_ANN

def load_history(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"History not found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_loss_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("ANN Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("📂 Loading training history...")
    history = load_history(HISTORY_PATH_ANN)

    print("📉 Plotting learning curves...")
    plot_loss_curves(history)
