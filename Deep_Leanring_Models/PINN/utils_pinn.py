import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

def load_data_and_prepare():
    df = pd.read_csv(config.TRAIN_FOR_MODEL_FILE)
    df = df.drop(columns=config.DROP_COLUMNS, errors='ignore')

    X = df.drop(columns=config.TARGET_COLUMNS)
    y = df[config.TARGET_COLUMNS]

    physics_keys = ['t_rise_est', 't_fall_est', 'f_ring_est', 'overshoot_norm_1', 'Vbus']
    physics_inputs = {k: X[k].values.astype("float32") for k in physics_keys}
    X_model = X.drop(columns=physics_keys)

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_scaled = input_scaler.fit_transform(X_model)
    y_scaled = output_scaler.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=config.SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=config.SEED)

    return X_train.astype("float32"), X_val.astype("float32"), X_test.astype("float32"), \
           y_train.astype("float32"), y_val.astype("float32"), y_test.astype("float32"), \
           input_scaler, output_scaler, physics_inputs

def inverse_transform_outputs(scaled_y, output_scaler):
    return output_scaler.inverse_transform(scaled_y)
