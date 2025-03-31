import os
import time
from datetime import datetime, timedelta
import gzip
import shutil
import subprocess
import sys
from tqdm import tqdm  # Import tqdm for progress bar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src import SMS_notifier

# Set up the start date and end date
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years ago

# Helper function to download and unzip the file
def download_and_unzip(ticker, date):
    # Format date as YYYY/MM
    year_month = date.strftime("%Y/%m")
    file_name = f"{date.strftime('%Y-%m-%d')}.csv.gz"
    
    # Specify a valid directory path, such as your Downloads folder
    file_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'VSCode Repos', 'trading_model', 'src', 'pre_processing', 'stock_data', year_month, file_name)

    # Make the directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Use mc to download the file from S3
    file_url = f"s3polygon/flatfiles/us_stocks_sip/day_aggs_v1/{date.year}/{date.month:02d}/{file_name}"
    result = subprocess.run(["mc", "cp", file_url, file_path], capture_output=True)

    # Check if the file was successfully downloaded
    if result.returncode != 0:
        return

    # Check if the file exists before unzipping
    if os.path.exists(file_path):
        # Unzip the file
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_path[:-3], 'wb') as f_out:  # remove .gz extension
                shutil.copyfileobj(f_in, f_out)
        # Remove the original gzipped file after unzipping
        os.remove(file_path)
    else:
        print(f"File {file_path} not found. Skipping unzip.")

def download_data():
    # Loop through each year and month from 5 years ago to today
    current_date = start_date
    total_days = (end_date - start_date).days  # Calculate the total number of days for the progress bar

    with tqdm(total=total_days, desc="Downloading data", unit="day") as pbar:
        while current_date < end_date:
            download_and_unzip("ticker", current_date)
            # Increment date by 1 day
            current_date += timedelta(days=1)
            pbar.update(1)  # Update the progress bar by 1 step

    SMS_notifier.send_sms_notification("S3 download script completed successfully.")

if __name__ == "__main__":
    download_data()    
    SMS_notifier.send_sms_notification("S3 download script completed successfully.")