import pandas as pd

# Update 'medium_articles.csv' to the actual name of your file if different
file_name = 'medium-english-50mb.csv' 

try:
    df = pd.read_csv(file_name)
    print(f"Columns in {file_name}:")
    print(df.columns.tolist())
    print("\nFirst row sample:")
    print(df.head(1))
except Exception as e:
    print(f"Error reading CSV: {e}")