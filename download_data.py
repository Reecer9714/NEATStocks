import os
import time
from datetime import date
from dateutil.relativedelta import relativedelta
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

for symbol in stock_symbols:
    file_name = f'data/{symbol}-intraday'
    file_path = os.path.join(local_dir, file_name)
    if not os.path.exists(file_path):
        print(f'Downloading {symbol} intraday data from Aplha Vantage...')
        try:
            data_price, _ = ts.get_daily(symbol, outputsize='full')
            data_rsi, _ = ti.get_rsi(symbol, interval='daily')
            data_ema, _ = ti.get_ema(symbol, interval='daily', time_period=10)
            data_sma, _ = ti.get_sma(symbol, interval='daily', time_period=50)
            data_obv, _ = ti.get_obv(symbol, interval='daily')
            df = DataFrame(data=data_price)
            daily_df = df.resample('D').last()
            daily_df.dropna(inplace=True)
            full_df = daily_df.join([
                data_rsi.resample('D').last(), 
                data_ema.resample('D').last(), 
                data_sma.resample('D').last(),
                data_obv.resample('D').last(),
            ])
            full_df.drop(columns='5. volume', inplace=True)
            full_df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                'RSI': 'rsi',
                'EMA': 'ema',
                'SMA': 'sma',
                'OBV': 'obv'},
                inplace=True)
            full_df.dropna(inplace=True)
            last_2_years = date.today() - relativedelta( years = +2 )
            full_df = full_df.loc[last_2_years.strftime('%Y-%m-%d'):date.today().strftime('%Y-%m-%d')]
            full_df.to_pickle(file_path)
            print(f'Downloaded {full_df.shape[0]} Days of data\n')

            #Wait till api free tier resets
            time.sleep(60)
        except ValueError as err:
            print(f'Error downloading: {err}')
