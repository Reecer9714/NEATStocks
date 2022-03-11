# NEATStocks

## Description
This project uses neat-python and stock data from Alpha Vantage to train a model to choose whether buy, sell, or hold a stock.

## Setup
Create python virtual environment ```python3 -m venv .env```  
Activate environment ```source .env/bin/activate```  
Install requirements ```pip install -r requirements.txt```  

Create secret.py with ```ALPHA_API_KEY="123456789XXXX"```

Download stock training data with [download_data.py](download_data.py)  
  This may take multiple tries to download data for all tickers in [config.py](config.py)
