import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from functools import lru_cache

class SPScraper:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.response = requests.get(self.base_url)
        self.response.raise_for_status()
        self.soup = BeautifulSoup(self.response.text, 'html.parser')
    
    @lru_cache(maxsize=1)
    def scrape_sp500_symbols(self):
        # Get current constituents table
        constituents_table = self.soup.find('table', {'id': 'constituents'})
        changes_table = self.soup.find('table', {'id': 'changes'})
        
        # Create DataFrame for current constituents
        current_symbols = []
        if constituents_table:
            rows = constituents_table.find_all('tr')[1:]
            for row in rows:
                cells = row.find_all('td')
                if cells:
                    symbol = cells[0].text.strip()
                    date_added = cells[5].text.strip() if len(cells) > 6 else None
                    current_symbols.append({
                        'symbol': symbol,
                        'date': date_added,
                    })
        
        # Create DataFrames
        current_df = pd.DataFrame(current_symbols)
        current_df.set_index('symbol', inplace=True)

        return current_df

if __name__ == "__main__":
    scraper = SPScraper()
    print(scraper.scrape_sp500_symbols())