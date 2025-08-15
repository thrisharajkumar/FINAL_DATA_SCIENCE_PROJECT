import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import TARGET_COLUMNS, DROP_COLUMNS, INPUT_SCALER_PATH_ANN, OUTPUT_SCALER_PATH_ANN

def preprocess_from_file(file_path):
    df = pd.read_csv(file_path).dropna()
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMNS]

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_scaled = input_scaler.fit_transform(X)
    y_scaled = output_scaler.fit_transform(y)

    joblib.dump(input_scaler, INPUT_SCALER_PATH_ANN)
    joblib.dump(output_scaler, OUTPUT_SCALER_PATH_ANN)

    return X_scaled, y_scaled, input_scaler, output_scaler, X.columns.tolist()

def preprocess_and_split_train(file_path):
    df = pd.read_csv(file_path).dropna()
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMNS]

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_scaled = input_scaler.fit_transform(X)
    y_scaled = output_scaler.fit_transform(y)

    joblib.dump(input_scaler, INPUT_SCALER_PATH_ANN)
    joblib.dump(output_scaler, OUTPUT_SCALER_PATH_ANN)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test, X.columns.tolist()

def load_and_scale_for_evaluation(file_path, input_scaler, output_scaler):
    df = pd.read_csv(file_path).dropna()
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET_COLUMNS].values
    X_scaled = input_scaler.transform(X)
    return X_scaled, y
