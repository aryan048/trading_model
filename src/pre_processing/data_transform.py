import os
import pandas as pd
from sqlalchemy import create_engine
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src import SMS_notifier

# SQLite connection string
database = 'stock_data.db'

# Create SQLAlchemy engine for connecting to SQLite
engine = create_engine(f'sqlite:///{database}')

def process_and_insert_files():
    
    # Now, connect to the SQLite database
    engine = create_engine(f'sqlite:///{database}')
    
    # Path to the stock data directory
    data_dir = 'src/pre_processing/stock_data'

    # Walk through the directory
    for year in os.listdir(data_dir):
        year_path = os.path.join(data_dir, year)
        if os.path.isdir(year_path):  # Check if it's a directory
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if os.path.isdir(month_path):  # Check if it's a directory
                    for file_name in os.listdir(month_path):
                        file_path = os.path.join(month_path, file_name)
                        if file_name.endswith('.csv'):
                            # Extract the date from the filename (assuming the filename format is 'YYYY-MM-DD.csv')
                            date_str = file_name.split('.')[0]  # Remove the '.csv' part
                            
                            try:
                                date = pd.to_datetime(date_str, format='%Y-%m-%d')  # Adjust format if needed
                            except Exception as e:
                                continue  # Skip this file if date parsing fails

                            # Read the CSV file into a DataFrame
                            df = pd.read_csv(file_path, delimiter=',', na_values=['', 'NULL'])  # Avoid treating "NAN" as NaN

                            if 'ticker' not in df.columns:
                                continue  # Skip files without a ticker column

                            df['ticker'] = df['ticker'].fillna('NAN').astype(str).str.upper()
                            
                            # Add the extracted date as a new column
                            df['date'] = date
                            
                            # Insert data into SQLite
                            df.to_sql('stock_data', engine, if_exists='append', index=False)
                            print(f"Inserted data from {file_path} into the database.")

    SMS_notifier.send_sms_notification("DB populated successfully")

if __name__ == '__main__':
    process_and_insert_files()
    SMS_notifier.send_sms_notification("DB populated successfully")