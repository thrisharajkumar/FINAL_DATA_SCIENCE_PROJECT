import os
import re
import sys
import pandas as pd
from PyPDF2 import PdfReader

import sys
import os

# Add the project root to sys.path to allow import of config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from config import DATA_EXTRACTION_INPUT, DATA_EXTRACTION_OUTPUT


# === Helper: Extract single value near a label ===
def find_value(text, label_pattern, context=100, dtype=float):
    try:
        pattern = rf"{label_pattern}.{{0,{context}}}?([\d.]+)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return dtype(match.group(1))
    except Exception:
        pass
    return None

# === Helper: Extract Min/Typ/Max values from tabular rows ===
def extract_values_by_label(text, label, expected_count=3):
    try:
        pattern = rf"{label}\s*\n?\s*((?:-?\d+\.?\d*\s+){{{expected_count - 1},}}[\d.+-]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            numbers = re.findall(r"-?\d+\.?\d*", match.group(1))
            return tuple(map(float, numbers[:expected_count])) + (None,) * (expected_count - len(numbers))
    except:
        pass
    return (None,) * expected_count

# === Helper: Extract nth occurrence of value ===
def extract_reverse_value(text, label, unit, occurrence=1):
    pattern = rf"{label}[^0-9A-Za-z]+([\d.]+)\s*{unit}"
    matches = re.findall(pattern, text, re.IGNORECASE)
    try:
        return float(matches[occurrence - 1])
    except:
        return None

# === Helper: Extract Tj_max from temperature block ===
def extract_temperature_max(text):
    match = re.search(
        r"Operating Junction and Storage Temperature[^+0-9\-]+[-–]?\d+\s*(?:to|–)\s*\+?(\d+)",
        text, re.IGNORECASE
    )
    return int(match.group(1)) if match else None

# === Main Processing Loop ===
for filename in os.listdir(DATA_EXTRACTION_INPUT):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(DATA_EXTRACTION_INPUT, filename)
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    device_id = filename.replace(".pdf", "")

    # === Feature Extraction ===
    VGS_th_min, VGS_th_typ, VGS_th_max = extract_values_by_label(text, r"Gate Threshold Voltage", expected_count=3)
    RDS_on_typ, RDS_on_max = extract_values_by_label(text, r"On-State Resistance", expected_count=2)[:2]
    Qg_total = extract_values_by_label(text, r"Total Gate Charge", expected_count=1)[0]

    Qgs = find_value(text, r"Gate to Source Charge[^0-9]*(\d+\.?\d*)")
    Qgd = find_value(text, r"Gate to Drain Charge[^0-9]*(\d+\.?\d*)")
    Qrr = extract_reverse_value(text, "Reverse Recovery Charge", "nC", occurrence=2)
    Irrm = extract_reverse_value(text, "Peak Reverse Recovery Current", "A", occurrence=2)
    Rth_typ, Rth_max = extract_values_by_label(text, r"Thermal Resistance from Junction to Case", expected_count=2)[:2]
    Tj_max = extract_temperature_max(text)

    data = {
        "DeviceID": device_id,
        "VBR_DSS": find_value(text, r"Drain - Source Voltage", dtype=int),
        "VGS_th_min": VGS_th_min,
        "VGS_th_typ": VGS_th_typ,
        "VGS_th_max": VGS_th_max,
        "RDS_on_typ": RDS_on_typ,
        "RDS_on_max": RDS_on_max,
        "Qg_total": Qg_total,
        "Qgs": Qgs,
        "Qgd": Qgd,
        "Rg_int": find_value(text, r"Internal Gate Resistance"),
        "Ciss": find_value(text, r"Input Capacitance", dtype=int),
        "Coss": find_value(text, r"Output Capacitance", dtype=int),
        "Crss": find_value(text, r"Reverse Transfer Capacitance", dtype=int),
        "td_on": find_value(text, r"Turn-On Delay Time"),
        "tr": find_value(text, r"Rise Time"),
        "td_off": find_value(text, r"Turn-Off Delay Time"),
        "tf": find_value(text, r"Fall Time"),
        "Eon": find_value(text, r"\bEON\b[^0-9]*(\d+\.?\d*)"),
        "Eoff": find_value(text, r"\bEOFF\b[^0-9]*(\d+\.?\d*)"),
        "Qrr": Qrr,
        "trr": find_value(text, r"Reverse Recovery Time[^a-zA-Z0-9]*(\d+)", dtype=int),
        "Irrm": Irrm,
        "Rth_JC_typ": Rth_typ,
        "Rth_JC_max": Rth_max,
        "Tj_max": Tj_max,
        "Power_dissipation": find_value(text, r"Power Dissipation", dtype=int)
    }

    df = pd.DataFrame([data])
    output_csv = os.path.join(DATA_EXTRACTION_OUTPUT, f"{device_id}.csv")
    df.to_csv(output_csv, index=False)
    print(f"[✓] Extracted: {output_csv}")