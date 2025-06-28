import requests
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import time

# Load API key from environment variable for security
API_KEY = '680d28928681c6.96535696'
if not API_KEY:
    raise ValueError("Please set the EODHD_API_KEY environment variable")

# File paths
SELECTED_STOCKS_FILE = "selected_stocks.json"
OUTPUT_FILE = "selected_stocks_with_dates.json"

def load_selected_stocks() -> Dict[str, Any]:
    """Load selected stocks from JSON file"""
    if os.path.exists(SELECTED_STOCKS_FILE):
        with open(SELECTED_STOCKS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_stocks_with_dates(stocks: Dict[str, Any]) -> None:
    """Save stocks with date information to JSON file"""
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(stocks, f, indent=2)
    print(f"Saved updated stocks data to {OUTPUT_FILE}")

def get_stock_date_range(ticker: str) -> Optional[Dict[str, str]]:
    """
    Fetch start and end dates for a given ticker using EODHD API
    
    Args:
        ticker (str): The stock ticker symbol
    
    Returns:
        Optional[Dict[str, str]]: Dictionary with 'Start date' and 'End date' or None if error
    """
    # Use the same API format as in eodhd_test.py
    url = f"https://eodhd.com/api/eod/{ticker}.US"
    
    params = {
        'api_token': API_KEY,
        'fmt': 'json'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print(f"No data found for ticker {ticker}")
            return None
        
        # Get first and last dates from the data
        dates = [item['date'] for item in data if 'date' in item]
        if not dates:
            print(f"No date information found for ticker {ticker}")
            return None
        
        start_date = min(dates)
        end_date = max(dates)
        
        return {
            "Start date": start_date,
            "End date": end_date
        }
        
    except Exception as e:
        print(f"Error fetching date range for {ticker}: {e}")
        return None

def process_stocks_with_dates():
    """Main function to process all stocks and add date information"""
    # Load existing stocks
    stocks = load_selected_stocks()
    
    if not stocks:
        print("No stocks found in selected_stocks.json")
        return
    
    print(f"Processing {len(stocks)} stocks...")
    
    # Process each stock
    for ticker, stock_data in stocks.items():
        print(f"Processing {ticker}...")
        
        # Get the original ticker code from the data
        original_ticker = stock_data.get('data', {}).get('Code')
        
        if not original_ticker:
            print(f"No 'Code' found for {ticker}, skipping...")
            continue
        
        # Get date range for this ticker
        date_range = get_stock_date_range(original_ticker)
        
        if date_range:
            # Add date information to the stock's data dictionary
            if 'data' not in stock_data:
                stock_data['data'] = {}
            
            stock_data['data'].update(date_range)
            print(f"  Added date range: {date_range['Start date']} to {date_range['End date']}")
        else:
            print(f"  Could not retrieve date range for {ticker}")
        
        # Add a small delay to avoid hitting API rate limits
        time.sleep(0.1)
    
    # Save the updated stocks data
    save_stocks_with_dates(stocks)
    print(f"\nCompleted processing {len(stocks)} stocks.")

if __name__ == "__main__":
    process_stocks_with_dates() 