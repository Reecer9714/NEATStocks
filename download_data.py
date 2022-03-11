import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from pandas import DataFrame
import secret
from config import stock_symbols

local_dir = os.path.dirname(__file__)

ts = TimeSeries(
    key=secret.ALPHA_API_KEY,
    output_format='pandas',
    indexing_type='date')
ti = TechIndicators(
    key=secret.ALPHA_API_KEY,
    output_format='pandas',
    indexing_type='date')

data_folder = os.path.join(local_dir, 'data')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

for symbol in stock_symbols:
    file_path = os.path.join(data_folder, f'{symbol}-intraday')
    if not os.path.exists(file_path):
        print(f'Downloading {symbol} intraday data from Aplha Vantage...')
        try:
            data_price, meta_data = ts.get_intraday(symbol, '60min', 'full')
            data_rsi, meta_data = ti.get_rsi(symbol, interval='daily')
            data_sma, meta_data = ti.get_sma(symbol, interval='daily')
            df = DataFrame(data=data_price)
            daily_df = df.resample('D').first()
            full_df = daily_df.join(
                [data_rsi.resample('D').first(), data_sma.resample('D').first()])
            full_df.drop(columns='5. volume', inplace=True)
            full_df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                'RSI': 'rsi',
                'SMA': 'sma'},
                inplace=True)
            full_df.to_pickle(file_path)
        except ValueError as err:
            print(f'Error downloading: {err}')
