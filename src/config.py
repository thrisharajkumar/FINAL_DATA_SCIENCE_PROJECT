# src/config.py

import os

# Root directory (project base)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_FILE = "data/processed/train_for_model.csv"
UNSEEN_HOLDOUT_FILE = "data/processed/unseen_holdout.csv"


# Data paths
TRAIN_FILE = os.path.join(ROOT_DIR, "data", "raw", "merged_train_with_features.csv")
TEST_FILE = os.path.join(ROOT_DIR, "data", "raw", "merged_test_with_features.csv")

# Target EMI output columns
TARGET_COLUMNS = [
    'voltage_rise_time_pulse1', 'voltage_rise_time_pulse2',
    'voltage_fall_time_pulse1', 'voltage_fall_time_pulse2',
    'current_rise_time_pulse1', 'current_rise_time_pulse2',
    'current_fall_time_pulse1', 'current_fall_time_pulse2',
    'overshoot_pulse_1', 'overshoot_pulse_2',
    'undershoot_pulse_1', 'undershoot_pulse_2',
    'ringing_frequency_MHz'
]

# Columns to drop during training
DROP_COLUMNS = ['DeviceID'] + TARGET_COLUMNS

# Output size
NUM_OUTPUTS = len(TARGET_COLUMNS)

# Artifacts
INPUT_SCALER_PATH = os.path.join(ROOT_DIR, "models", "input_scaler.pkl")
OUTPUT_SCALER_PATH = os.path.join(ROOT_DIR, "models", "output_scaler.pkl")
BEST_PARAMS_PATH = os.path.join(ROOT_DIR, "models", "best_params.pkl")
FINAL_MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.h5")
FEATURES_JSON_PATH = os.path.join(ROOT_DIR, "data", "processed", "selected_features.json")

# For reproducibility
SEED = 42
