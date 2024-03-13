import pandas as pd

# Load the Excel file
excel_data = pd.read_excel('./carbon cathode.xlsx')

# Convert the Excel file to CSV, ensuring the first row is the header
csv_path = 'model/carbon cathode.csv'
excel_data.to_csv(csv_path, index=False)
