import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import (
    SIMULATION_DATA_DIR,
    DATASHEET_EXTRACTED_DIR,
    MERGED_OUTPUT_DIR
)

mosfet_names = ["C2M0025120D", "C2M0040120D", "C2M0080120D", 
                "C2M0160120D", "C2M0280120D", "C2M1000170D"]  

os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

datasheet_order = [
    "DeviceID", "VBR_DSS", "VGS_th_min", "VGS_th_max", "RDS_on_typ", "RDS_on_max",
    "Qg_total", "Qgs", "Qgd", "Rg_int", "Ciss", "Coss", "Crss",
    "td_on", "tr", "td_off", "tf", "Eon", "Eoff", "Qrr", "trr", "Irrm",
    "Rth_JC_typ", "Rth_JC_max", "Tj_max", "Power_dissipation"
]

for mosfet in mosfet_names:
    try:
        sim_file = os.path.join(SIMULATION_DATA_DIR, f"simulation_results_{mosfet}.csv")
        ds_file = os.path.join(DATASHEET_EXTRACTED_DIR, f"datasheet_{mosfet}.csv")
        out_file = os.path.join(MERGED_OUTPUT_DIR, f"merged_{mosfet}.csv")

        simulation_df = pd.read_csv(sim_file)
        datasheet_df = pd.read_csv(ds_file)
        datasheet_row = datasheet_df.iloc[0][datasheet_order]

        for col in reversed(datasheet_order):
            simulation_df.insert(0, col, datasheet_row[col])

        simulation_df.to_csv(out_file, index=False)
        print(f"Merged {mosfet} → {out_file}")

    except Exception as e:
        print(f"Failed to process {mosfet}: {e}")
