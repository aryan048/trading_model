import requests
from bs4 import BeautifulSoup

def scrape_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})  # Find the table with id 'constituents'

    symbols = []
    if table:
        rows = table.find_all('tr')[1:]  # Skip the header row
        for row in rows:
            symbol_cell = row.find_all('td')[0]  # The first column contains the symbol
            if symbol_cell:
                symbols.append(symbol_cell.text.strip())  # Extract and clean the symbol text

    return symbols

if __name__ == "__main__":
    symbols = scrape_sp500_symbols()
    print(symbols)
