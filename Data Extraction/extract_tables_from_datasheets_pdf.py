import os
import pdfplumber
import pandas as pd
import re

# Define folders
input_folder = "SICMOSFETS"
output_folder = "SICMOSFET_CSV_FILES"
os.makedirs(output_folder, exist_ok=True)

# Section titles to extract
target_keywords = [
    "maximum ratings", "electrical characteristics",
    "reverse diode", "thermal characteristics"
]

def clean_line_encoding(line):
    """Fix character encoding issues (e.g., degree symbol)."""
    return line.replace("ËšC", "°C").replace("˚C", "°C").replace("Â°C", "°C")

def clean_dataframe(df):
    """Remove newlines and normalize whitespace in headers and cells."""
    # Clean column names
    df.columns = [re.sub(r"\s+", "", str(col)) for col in df.columns]

    # Clean each cell using .apply + .map
    df = df.apply(lambda col: col.map(lambda x: re.sub(r"\s+", " ", str(x)).strip() if pd.notnull(x) else x))
    return df

def find_matching_heading(text_lines):
    """Return heading line if it contains any of the target keywords."""
    for line in text_lines:
        cleaned = clean_line_encoding(line).strip().lower()
        for keyword in target_keywords:
            if keyword in cleaned:
                return clean_line_encoding(line.strip())
    return None

def standardize_columns(df):
    """Fallback column names if header extraction fails."""
    df.columns = [f"Col{i+1}" for i in range(len(df.columns))]
    return df

# Process each PDF file
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename.replace(".pdf", ".csv"))
    collected_tables = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text_lines = page.extract_text().split("\n") if page.extract_text() else []
                tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2 or len(table[0]) < 2:
                        continue

                    heading = find_matching_heading(text_lines)
                    if heading:
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0])
                        except Exception:
                            df = pd.DataFrame(table)
                            df = standardize_columns(df)

                        # Clean header and cell values
                        df = clean_dataframe(df)

                        # Clean degree symbols
                        df = df.apply(lambda col: col.map(lambda x: clean_line_encoding(str(x)) if pd.notnull(x) else x))

                        collected_tables.append((heading, df))

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

    # Save cleaned CSV
    if collected_tables:
        all_csv_data = []
        for title, df in collected_tables:
            all_csv_data.append(pd.DataFrame([[title]]))  # Section title row
            df = standardize_columns(df)
            all_csv_data.append(df)
            all_csv_data.append(pd.DataFrame([[""] * len(df.columns)], columns=df.columns))  # Blank row

        final_df = pd.concat(all_csv_data, ignore_index=True)
        final_df.to_csv(output_path, index=False, header=False)
        print(f"Extracted: {filename} → {output_path}")
    else:
        print(f"No matching tables found in {filename}")
