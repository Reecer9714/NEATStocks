from enum import Enum
import os
import random
import pandas as pd
from neat.math_util import softmax, mean, stdev

from config import *

class Action(Enum):
    HOLD = 0
    SELL = 1
    BUY = 2

class StockSimulator:
    def __init__(self) -> None:
        self.reset_sim()
        self.stock_data = pd.DataFrame()
        pass

    def load_data(self, symbol):
        self.stock_data = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'data/{}-intraday'.format(symbol)))
        return self

    def reset_sim(self):
        self.cash = starting_money
        self.position = 0
        self.bought_stocks = []
        self.actions = []
        self.losses = []
        self.last_buy = 0
        self.last_sell = 0
        self.current_stock_value = 0

    def buy_stock(self, price):
        if price * postitions_to_buysell <= self.cash:
            self.cash -= price * postitions_to_buysell
            self.position += postitions_to_buysell
            self.bought_stocks.append(price)
            self.last_buy = self.bought_stocks[0]
            return price
        else:
            return -1

    def sell_stock(self, price):
        if self.position >= postitions_to_buysell:
            self.cash += price * postitions_to_buysell
            self.position -= postitions_to_buysell
            self.last_sell = price
            buy_price = self.bought_stocks.pop(0)
            profit_pct = price / buy_price
            if profit_pct < 1:
                self.losses.append(profit_pct)
            return profit_pct
        else:
            return -1

    def sim_strategy(self, strategy, starting_index, sim_length):
        for day_index in range(starting_index,starting_index+sim_length):
            day_data = self.stock_data.iloc[day_index].values.tolist()
            day_data.extend( self.stock_data.iloc[day_index-1].values.tolist() )
            day_data.extend([self.cash, self.position, self.last_buy, self.last_sell])

            class_output = strategy(day_data)

            close_price = random.uniform(day_data[Inputs.OPEN.value],day_data[Inputs.CLOSE.value])
            if class_output == Action.SELL:
                profit_pct = self.sell_stock(close_price)
                if profit_pct >= 0:
                    self.actions.append((day_index, "Sell", close_price, profit_pct))
            elif class_output == Action.BUY:
                price = self.buy_stock(close_price)
                if price >= 0:
                    self.actions.append((day_index, "Buy", close_price, close_price))
            self.current_stock_value = self.position * close_price

    def evaluate(self):  
        if len(self.losses) > 0:
            avg_loss = mean(self.losses)
        else:
            avg_loss = 0.9 # Assume 10% loss
        overal_profit = (self.current_stock_value * avg_loss + self.cash) / starting_money
        if self.last_buy == 0:
            overal_profit = 0
        return overal_profit, avg_loss

    def stats(self):
        stats = {
            "sells": {},
            "losses": {}
        }
        sells = [action[3] for action in self.actions if action[1] == "Sell"]
        stats["sells"]['count'] = len(sells)
        if len(sells) > 0:
            stats["sells"]['min'] = min(sells)
            stats["sells"]['max'] = max(sells)
            stats["sells"]['mean'] = mean(sells)
            stats["sells"]['stdev'] = stdev(sells)
        losses = [sell for sell in sells if sell < 1]
        stats["losses"]['count'] = len(losses)
        if len(losses) > 0:
            stats["losses"]['min'] = min(losses)
            stats["losses"]['max'] = max(losses)
            stats["losses"]['mean'] = mean(losses)
            stats["losses"]['stdev'] = stdev(losses)
        return stats
