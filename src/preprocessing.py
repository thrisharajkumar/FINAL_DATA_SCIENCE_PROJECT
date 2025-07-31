import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COLUMNS, DROP_COLUMNS, INPUT_SCALER_PATH, OUTPUT_SCALER_PATH

def preprocess_and_split_train(file_path):
    df = pd.read_csv(file_path).dropna()
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMNS]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(X_train)
    X_val_scaled = input_scaler.transform(X_val)
    X_test_scaled = input_scaler.transform(X_test)

    y_train_scaled = output_scaler.fit_transform(y_train)
    y_val_scaled = output_scaler.transform(y_val)
    y_test_scaled = output_scaler.transform(y_test)

    joblib.dump(input_scaler, INPUT_SCALER_PATH)
    joblib.dump(output_scaler, OUTPUT_SCALER_PATH)

    return (X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled,
            X.columns.tolist())

def preprocess_holdout_test(holdout_path):
    import json
    with open("models/input_features.json", "r") as f:
        feature_names = json.load(f)

    df = pd.read_csv(holdout_path).dropna()
    X = df[feature_names]
    y = df[TARGET_COLUMNS]

    input_scaler = joblib.load(INPUT_SCALER_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)

    X_scaled = input_scaler.transform(X)
    y_scaled = output_scaler.transform(y)

    return X_scaled, y_scaled, y.values
