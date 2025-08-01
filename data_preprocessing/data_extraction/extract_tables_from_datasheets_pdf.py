import os
import sys
import pdfplumber
import pandas as pd
import re

# === Add root to sys.path to import config ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from config import DATA_TABLES_INPUT, DATA_TABLES_OUTPUT

# === Create output directory ===
os.makedirs(DATA_TABLES_OUTPUT, exist_ok=True)

# === Target section titles to extract from datasheets ===
target_keywords = [
    "maximum ratings", "electrical characteristics",
    "reverse diode", "thermal characteristics"
]

def clean_line_encoding(line):
    return line.replace("ËšC", "°C").replace("˚C", "°C").replace("Â°C", "°C")

def clean_dataframe(df):
    df.columns = [re.sub(r"\s+", "", str(col)) for col in df.columns]
    return df.apply(lambda col: col.map(lambda x: re.sub(r"\s+", " ", str(x)).strip() if pd.notnull(x) else x))

def find_matching_heading(text_lines):
    for line in text_lines:
        cleaned = clean_line_encoding(line).strip().lower()
        for keyword in target_keywords:
            if keyword in cleaned:
                return clean_line_encoding(line.strip())
    return None

def standardize_columns(df):
    df.columns = [f"Col{i+1}" for i in range(len(df.columns))]
    return df

# === Process each PDF ===
for filename in os.listdir(DATA_TABLES_INPUT):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(DATA_TABLES_INPUT, filename)
    output_path = os.path.join(DATA_TABLES_OUTPUT, filename.replace(".pdf", ".csv"))
    collected_tables = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_lines = page.extract_text().split("\n") if page.extract_text() else []
                tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2 or len(table[0]) < 2:
                        continue

                    heading = find_matching_heading(text_lines)
                    if heading:
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0])
                        except:
                            df = pd.DataFrame(table)
                            df = standardize_columns(df)

                        df = clean_dataframe(df)
                        df = df.apply(lambda col: col.map(lambda x: clean_line_encoding(str(x)) if pd.notnull(x) else x))
                        collected_tables.append((heading, df))

    except Exception as e:
        print(f"[✗] Error processing {filename}: {e}")
        continue

    # === Save to CSV ===
    if collected_tables:
        all_csv_data = []
        for title, df in collected_tables:
            all_csv_data.append(pd.DataFrame([[title]]))  # section heading
            df = standardize_columns(df)
            all_csv_data.append(df)
            all_csv_data.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))  # spacing row

        final_df = pd.concat(all_csv_data, ignore_index=True)
        final_df.to_csv(output_path, index=False, header=False)
        print(f"[✓] Extracted: {filename} → {output_path}")
    else:
        print(f"[i] No matching tables found in {filename}")
