import json
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("data/raw/merged_train_with_features.csv")
test = pd.read_csv("data/raw/merged_test_with_features.csv")

with open("models/input_features.json", "r") as f:
    input_features = json.load(f)

for col in input_features:
    plt.figure(figsize=(6, 3))
    plt.hist(train[col], bins=100, alpha=0.5, label='Train', density=True)
    plt.hist(test[col], bins=100, alpha=0.5, label='Test', density=True)
    plt.title(f"{col} distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
