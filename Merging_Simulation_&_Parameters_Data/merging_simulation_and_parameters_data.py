import os
import pandas as pd

# === Configuration ===
mosfet_names = ["C2M0025120D", "C2M0040120D", "C2M0080120D", 
                "C2M0160120D", "C2M0280120D", "C2M1000170D"]  
datasheet_dir = "SiC_mosfet_datasheets"
simulation_dir = "Simulation_Data"
output_dir = "merged_data"
os.makedirs(output_dir, exist_ok=True)

# === Desired datasheet column order ===
datasheet_order = [
    "DeviceID", "VBR_DSS", "VGS_th_min", "VGS_th_max", "RDS_on_typ", "RDS_on_max",
    "Qg_total", "Qgs", "Qgd", "Rg_int", "Ciss", "Coss", "Crss",
    "td_on", "tr", "td_off", "tf", "Eon", "Eoff", "Qrr", "trr", "Irrm",
    "Rth_JC_typ", "Rth_JC_max", "Tj_max", "Power_dissipation"
]

# === Batch merging ===
for mosfet in mosfet_names:
    try:
        sim_file = os.path.join(simulation_dir, f"simulation_results_{mosfet}.csv")
        ds_file = os.path.join(datasheet_dir, f"datasheet_{mosfet}.csv")
        out_file = os.path.join(output_dir, f"merged_{mosfet}.csv")

        simulation_df = pd.read_csv(sim_file)
        datasheet_df = pd.read_csv(ds_file)
        datasheet_row = datasheet_df.iloc[0][datasheet_order]

        # Insert datasheet fields at front
        for col in reversed(datasheet_order):
            simulation_df.insert(0, col, datasheet_row[col])

        simulation_df.to_csv(out_file, index=False)
        print(f"Merged {mosfet} → {out_file}")

    except Exception as e:
        print(f"Failed to process {mosfet}: {e}")
