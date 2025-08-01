import pandas as pd
import numpy as np


def add_derived_features(df):
    """
    Adds physics-informed derived features for EMI modeling.
    Inputs:
        df - DataFrame containing MOSFET datasheet, simulation inputs and simulation results
    Returns:
        df - DataFrame with new derived features appended
    """

    df = df.copy()

    # === From "Minimizing Parasitic Effects in SiC MOSFET Modules" ===
    L_cols = ['Ls4', 'Ls5', 'Ls6', 'Ls7', 'Ls8', 'Ls9', 'Ls10']
    df['L_total'] = df[L_cols].sum(axis=1)

    # === From "Power MOSFET Basics" and "Gate Drive Circuits Guide" ===
    # Gate drive current (Ig = (Vdrv - Vth) / (Rg + Rg_int)), assume Vdrv = 12V
    df['gate_drive_current'] = (12 - df['VGS_th_max']) / (df['Rg'] + df['Rg_int'])

    # === From "Understanding and Predicting Power MOSFET Switching Behaviour" ===
    # Miller delay time = Qgd / Ig
    df['miller_delay'] = df['Qgd'] / df['gate_drive_current']

    # === From "Minimizing Parasitic Effects in SiC MOSFET Modules" ===
    # Resonant ringing frequency: f = 1 / (2π√(L_total × Coss))
    df['f_ring_est'] = 1 / (2 * np.pi * np.sqrt(df['L_total'] * df['Coss']))

    # === From "Power MOSFET Basics" ===
    # Normalized overshoot to Vbus
    df['overshoot_norm_1'] = df['overshoot_pulse_1'] / df['Vbus']
    df['overshoot_norm_2'] = df['overshoot_pulse_2'] / df['Vbus']

    # Gate charge and capacitance ratios
    df['Qgd_Qgs_ratio'] = df['Qgd'] / df['Qgs']
    df['Qg_Ciss_ratio'] = df['Qg_total'] / df['Ciss']
    df['Cgd_Ciss_ratio'] = df['Crss'] / df['Ciss']
    df['Coss_Ciss_ratio'] = df['Coss'] / df['Ciss']

    # === From "Power MOSFET Basics" ===
    # Estimated rise/fall time: (Rg + Rg_int) * capacitance
    df['t_rise_est'] = (df['Rg'] + df['Rg_int']) * df['Ciss']
    df['t_fall_est'] = (df['Rg'] + df['Rg_int']) * df['Crss']

    # === From "Design and Application Guide for High Speed MOSFET Gate Drive Circuits" ===
    # Switching energy: E = 0.5 * Vbus * Ig * (tr + tf)
    df['E_sw_est'] = 0.5 * df['Vbus'] * df['gate_drive_current'] * (df['t_rise_est'] + df['t_fall_est'])

    # === From "Understanding Diode Reverse Recovery" ===
    # Diode reverse recovery loss estimate (P = Qrr * Vbus * f_sw), assuming f_sw = 100kHz
    df['P_rr_est'] = df['Qrr'] * df['Vbus'] * 100_000

    # Derived ratio to measure diode quality
    df['Qrr_Irrm_ratio'] = df['Qrr'] / df['Irrm']

    # Thermal performance indicator
    df['Rth_ratio'] = df['Rth_JC_max'] / df['Rth_JC_typ']

    return df

import config

# Load your CSVs using config path
train_df = pd.read_csv(config.MERGED_TRAIN_FILE)

# Add derived features
train_df = add_derived_features(train_df)

# Save the enhanced dataset using config path
train_df.to_csv(config.MERGED_TRAIN_WITH_FEATURES_FILE, index=False)
