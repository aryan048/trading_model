import requests
import os
import json
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load API key from environment variable for security
API_KEY = '680d28928681c6.96535696'
if not API_KEY:
    raise ValueError("Please set the EODHD_API_KEY environment variable")

# File to store selected stocks
SELECTED_STOCKS_FILE = "selected_stocks.json"

def load_selected_stocks() -> Dict[str, Any]:
    """Load selected stocks from JSON file"""
    if os.path.exists(SELECTED_STOCKS_FILE):
        with open(SELECTED_STOCKS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_selected_stocks(stocks: Dict[str, Any]) -> None:
    """Save selected stocks to JSON file, merging with existing stocks"""
    # Load existing stocks
    existing_stocks = load_selected_stocks()
    # Merge new stocks with existing ones
    merged_stocks = {**existing_stocks, **stocks}
    # Save merged stocks
    with open(SELECTED_STOCKS_FILE, 'w') as f:
        json.dump(merged_stocks, f, indent=2)

def get_exchange_symbols(exchange_code: str) -> Dict[str, Any]:
    """
    Fetch exchange symbols from EODHD API
    
    Args:
        exchange_code (str): The exchange code (e.g., 'US', 'NYSE', 'NASDAQ')
    
    Returns:
        Dict[str, Any]: Dictionary mapping company names to their full data
    """
    base_url = "https://eodhd.com/api/exchange-symbol-list"
    url = f"{base_url}/{exchange_code}"
    
    params = {
        'api_token': API_KEY,
        'delisted': 1,
        'fmt': 'json'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Create dictionary with company names as keys
        company_data = {}
        for item in data:
            company_name = item.get('Name')
            if company_name:
                company_data[company_name] = item
        
        return company_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {}

def get_historical_data(ticker: str, exchange: str) -> pd.DataFrame:
    """
    Fetch historical data for a given ticker and exchange
    
    Args:
        ticker (str): The stock ticker symbol
        exchange (str): The exchange code
    
    Returns:
        pd.DataFrame: DataFrame containing historical price data
    """
    url = f"https://eodhd.com/api/eod/{ticker}.US?api_token={API_KEY}&fmt=json"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def plot_historical_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Plot historical price data
    
    Args:
        df (pd.DataFrame): DataFrame containing historical price data
        ticker (str): The stock ticker symbol
    """
    if df.empty:
        print("No data to plot")
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['adjusted_close'], label='AdjustedClose Price')
    plt.title(f'{ticker} Historical Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def search_companies(company_data: Dict[str, Any]) -> None:
    """
    Interactive function to search companies and view their data
    """
    # Load existing selected stocks
    selected_stocks = load_selected_stocks()
    
    print("\nEnter company name to search (or 'q' to quit):")
    while True:
        user_input = input("> ").strip().upper()
        
        if user_input.lower() == 'q':
            break
            
        # Find all companies that contain the search string and are Common Stock
        matching_companies = {
            name: data 
            for name, data in company_data.items() 
            if user_input in name.upper() and data.get('Type') == 'Common Stock'
        }
        
        if matching_companies:
            print("\nResults:")
            print("-" * 100)
            company_list = []
            for i, (name, data) in enumerate(matching_companies.items(), 1):
                print(f"{i}. {name}")
                print("   Data:")
                print(json.dumps(data, indent=2))
                print("-" * 100)
                company_list.append((name, data))
            
            print(f"Found {len(matching_companies)} matching companies")
            
            # Allow user to select a company
            while True:
                selection = input("\nEnter number to select a company (or press Enter to search again): ").strip()
                if not selection:
                    break
                    
                try:
                    # Check if selection ends with "Test"
                    if selection.lower().endswith("test"):
                        idx = int(selection[:-4]) - 1
                        if 0 <= idx < len(company_list):
                            selected_name, selected_data = company_list[idx]
                            ticker = selected_data.get('Code')
                            exchange = selected_data.get('Exchange')
                            
                            print(f"\nFetching historical data for {ticker}...")
                            df = get_historical_data(ticker, exchange)
                            if not df.empty:
                                plot_historical_data(df, ticker)
                            break
                    else:
                        idx = int(selection) - 1
                        if 0 <= idx < len(company_list):
                            selected_name, selected_data = company_list[idx]
                            ticker = selected_data.get('Code')
                            exchange = selected_data.get('Exchange')
                            
                            # Ask for new ticker
                            new_ticker = input("\nEnter new ticker for this stock: ").strip().upper()
                            if not new_ticker:
                                print("No new ticker provided. Skipping...")
                                break
                            
                            # Store the stock data with new ticker as key
                            selected_stocks[new_ticker] = {
                                'ticker': ticker,
                                'name': selected_name,
                                'exchange': exchange,
                                'data': selected_data
                            }
                            
                            # Save to JSON file
                            save_selected_stocks(selected_stocks)
                            print(f"\nAdded {new_ticker} to selected stocks. Total stocks: {len(selected_stocks)}")
                            break
                        else:
                            print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("\nNo matching companies found")

# Example usage
if __name__ == "__main__":
    print("Fetching stock data from US stocks")
    company_data = get_exchange_symbols("US")
    print(f"\nTotal companies: {len(company_data)}")
    search_companies(company_data)
