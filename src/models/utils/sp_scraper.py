import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from functools import lru_cache
import webbrowser
import tempfile
import os

class SPScraper:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.response = requests.get(self.base_url)
        self.response.raise_for_status()
        self.soup = BeautifulSoup(self.response.text, 'html.parser')
    
    @lru_cache(maxsize=1)
    def scrape_sp500_symbols(self)-> pd.DataFrame:
        # Get current constituents table
        constituents_table = self.soup.find('table', {'id': 'constituents'})
        # Get changes table
        changes_table = self.soup.find('table', {'id': 'changes'})
        
        # Create DataFrame for current constituents
        constituents = []
        if constituents_table:
            rows = constituents_table.find_all('tr')[1:]
            for row in rows:
                cells = row.find_all('td')
                if cells:
                    ticker = cells[0].text.strip()
                    security = cells[1].text.strip()
                    date = pd.to_datetime(cells[5].text.strip())
                    constituents.append({
                        'ticker_added': ticker,
                        'security_added': security,
                        'date': date,
                    })
        
        # Create DataFrames
        constituents_df = pd.DataFrame(constituents)
        constituents_df.set_index('date', inplace=True)

        changes = []
        if changes_table:
            rows = changes_table.find_all('tr')[1:]
            for row in rows:
                cells = row.find_all('td')
                if cells:
                    date = pd.to_datetime(cells[0].text.strip())
                    ticker_added = cells[1].text.strip()
                    security_added = cells[2].text.strip()
                    ticker_removed = cells[3].text.strip()
                    security_removed = cells[4].text.strip()
                    changes.append({
                        'date': date,
                        'ticker_added': ticker_added,
                        'security_added': security_added,
                        'ticker_removed': ticker_removed,
                        'security_removed': security_removed,
                    })

        changes_df = pd.DataFrame(changes)
        changes_df.set_index('date', inplace=True)

        # Concatenate on date index and handle duplicates
        stacked_df = pd.concat([constituents_df, changes_df], axis=0)
        stacked_df = stacked_df.sort_index(ascending=False)
        return constituents_df, changes_df, stacked_df
    

    def build_table(self, date: str)-> pd.DataFrame:
        """
        Builds a table of the S&P 500 constituents at a given date, works backwards from current constituents
        
        Args:
            date (str or datetime): The target date to get constituents for
            
        Returns:
            pd.DataFrame: DataFrame containing ticker for constituents at the given date
        """
        constituents_df, changes_df, historical_df = self.scrape_sp500_symbols()
        
        # Convert input date to datetime if it's a string
        date = pd.to_datetime(date).normalize()

            
        # Start with current constituents (only tickers)
        current_constituents = set(constituents_df['ticker_added'])
        
        # Walk through historical data in reverse chronological order (newest to oldest)
        for idx, row in historical_df.sort_index(ascending=False).iterrows():
            idx_normalized = pd.to_datetime(idx).normalize()


            # If we've gone past our target date, stop processing
            if idx_normalized <= date:
                break
                
            # When going backwards:
            # - If a stock was added, we need to remove it (it wasn't in the index at the target date)
            # - If a stock was removed, we need to add it back (it was in the index at the target date)
            if pd.notna(row['ticker_added']):
                current_constituents.discard(row['ticker_added'])
                
            if pd.notna(row['ticker_removed']):
                current_constituents.add(row['ticker_removed'])
        
        # Convert set to DataFrame
        if not current_constituents:
            return pd.DataFrame(columns=['ticker'])
            
        return pd.DataFrame({'ticker': list(current_constituents)})


if __name__ == "__main__":
    scraper = SPScraper()
    print(scraper.build_table('1995-03-10'))
