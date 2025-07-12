import pdfplumber
import pandas as pd

# trying to extract the tables from the page 1 in the datasheet for the MOSFET
# Loading the PDF and select the first page 
pdf_path = "file.pdf"

with pdfplumber.open(pdf_path) as pdf:
    first_page = pdf.pages[0]
    
    # Extract tables from the first page
    tables = first_page.extract_tables()

    # all tables to check which one is "Maximum Ratings"
    for idx, table in enumerate(tables):
        print(f"\n--- Table {idx+1} ---")
        for row in table:
            print(row)


mosfet_table = tables[0]
# secod table is usually the maximum ratings one
max_ratings_table = tables[1]


# saving to a CSV FILE 
df = pd.DataFrame(max_ratings_table[1:], columns=max_ratings_table[0])
df.to_excel("maximum_ratings.xlsx", index=False)

print("Maximum Ratings table saved to: maximum_ratings.xlsx")