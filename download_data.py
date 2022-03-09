import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from pandas import DataFrame
import secret
from config import stock_symbols

local_dir = os.path.dirname(__file__)

ts = TimeSeries(key=secret.ALPHA_API_KEY,
                output_format='pandas', indexing_type='integer')

for symbol in stock_symbols:
    data, meta_data = ts.get_intraday(symbol, '60min', 'full')
    df = DataFrame(data=data)
    df.to_pickle(os.path.join(local_dir, f'data/{symbol}-intraday'))
