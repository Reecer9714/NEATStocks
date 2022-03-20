from enum import Enum

config_file = 'stock-feedforward'
stock_symbols = ["FB", "AMD", "MSFT", "NFLX", "PYPL"]
starting_money = 5000
num_of_days_to_sim = 5 * 20
num_of_days_to_lookback = 0
postitions_to_buysell = 5

node_names = {
        -1: 'open', -2: 'high', -3: 'low', -4: 'close',
        -5: 'rsi', -6: 'ema', -7: 'sma', -8: 'slowd', -9: 'slowk',
        -10: 'last_open', -11: 'last_high', -12: 'last_low', -13: 'last_close',
        -14: 'last_rsi', -15: 'last_ema', -16: 'last_sma', -17: 'last_slowd', -18: 'last_slowk',
        -19: 'cash', -20: 'pos', -21: 'last_buy', -22: 'last_sell',
        0: 'hold', 1: 'sell', 2: 'buy', 3: 'amount'}

class Inputs(Enum):
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    RSI = 4
    EMA = 5
    SMA = 6
    SLOWD = 7
    SLOWK = 8
    LAST_OPEN = 9
    LAST_HIGH = 10
    LAST_LOW = 11
    LAST_CLOSE = 12
    LAST_RSI = 13
    LAST_EMA = 14
    LAST_SMA = 15
    LAST_SLOWD = 16
    LAST_SLOWK = 17
    CASH = 18
    POSITION = 19
    LAST_BUY = 20
    LAST_SELL = 21
