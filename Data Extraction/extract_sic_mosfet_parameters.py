import re
import os
import pandas as pd
from PyPDF2 import PdfReader

# === CONFIG ===
input_folder = "SICMOSFETS"
output_folder = "SICMOSFETS_Extracted_parameters"
os.makedirs(output_folder, exist_ok=True)

# === Extraction helper ===
def find_value(text, label, context=100, dtype=float):
    try:
        pattern = rf"{label}.{{0,{context}}}?([\d.]+)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return dtype(match.group(1))
    except:
        pass
    return None

# === Loop through PDF files ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_folder, filename)
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)

    device_id = filename.replace(".pdf", "")

    data = {
        "DeviceID": device_id,
        "VBR_DSS": find_value(text, "Drain - Source Voltage", dtype=int),
        "VGS_th_min": find_value(text, "Gate Threshold Voltage.*?(\d+\.?\d*)", dtype=float),
        "VGS_th_max": find_value(text, "Gate Threshold Voltage.*?\n.*?\n(\d+\.?\d*)", dtype=float),
        "RDS_on_typ": find_value(text, "On-State Resistance.*?(\d+)", dtype=float),
        "RDS_on_max": find_value(text, "On-State Resistance.*?\n\d+\n(\d+)", dtype=float),
        "Qg_total": find_value(text, "Total Gate Charge", dtype=float),
        "Qgs": find_value(text, "Gate to Source Charge", dtype=float),
        "Qgd": find_value(text, "Gate to Drain Charge", dtype=float),
        "Rg_int": find_value(text, "Internal Gate Resistance", dtype=float),
        "Ciss": find_value(text, "Input Capacitance", dtype=int),
        "Coss": find_value(text, "Output Capacitance", dtype=int),
        "Crss": find_value(text, "Reverse Transfer Capacitance", dtype=int),
        "td_on": find_value(text, "Turn-On Delay Time", dtype=float),
        "tr": find_value(text, "Rise Time", dtype=float),
        "td_off": find_value(text, "Turn-Off Delay Time", dtype=float),
        "tf": find_value(text, "Fall Time", dtype=float),
        "Eon": find_value(text, "EON", dtype=float),
        "Eoff": find_value(text, "EOFF", dtype=float),
        "Qrr": find_value(text, "Reverse Recovery Charge[^0-9]*(\d+)", dtype=int),
        "trr": find_value(text, "Reverse Recovery Time[^0-9]*(\d+)", dtype=int),
        "Irrm": find_value(text, "Peak Reverse Recover[y|y Current]*[^0-9]*(\d+)", dtype=int),
        "Rth_JC_typ": find_value(text, "Thermal Resistance from Junction to Case.*?\n(\d+\.\d+)", dtype=float),
        "Rth_JC_max": find_value(text, "Thermal Resistance from Junction to Case.*?\n\d+\.\d+\n(\d+\.\d+)", dtype=float),
        "Tj_max": find_value(text, "Storage Temperature.*?(\+?\d+)", dtype=int),
        "Power_dissipation": find_value(text, "Power Dissipation", dtype=int)
    }

    df = pd.DataFrame([data])
    output_csv = os.path.join(output_folder, f"{device_id}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Extracted: {output_csv}")
