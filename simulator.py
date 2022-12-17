from enum import Enum
from math import floor
from operator import mod
from os import path
import pandas as pd
from neat.math_util import mean, stdev

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
        self.stock_data = pd.read_pickle(path.join(path.dirname(__file__), 'data/{}-intraday'.format(symbol)))
        return self

    def reset_sim(self):
        self.cash = starting_money
        self.position = 0
        self.bought_stocks = []
        self.actions = []
        self.losses = []
        self.last_sell = 0
        self.current_stock_value = 0

    def buy_stock(self, price, amount):
        if amount > 0 and price * amount <= self.cash:
            self.cash -= price * amount
            self.position += amount
            self.bought_stocks.extend([price]*amount)
            return price
        else:
            return -1

    def sell_stock(self, price, amount):
        if amount > 0 and self.position >= amount:
            stocks_to_sell = self.bought_stocks[:amount]
            buy_price = mean(stocks_to_sell)
            self.bought_stocks = self.bought_stocks[amount:]
            self.cash += price * amount
            self.position -= amount
            self.last_sell = price
            profit_pct = price / buy_price
            if profit_pct < 1:
                self.losses.append(profit_pct)
            return profit_pct
        else:
            return -1

    def sim_strategy(self, strategy, starting_index, sim_length):
        for day_index in range(starting_index,starting_index+sim_length):
            if day_index % 7:
                self.cash += money_per_week
                pass
            day_data = self.stock_data.iloc[day_index-1].values.tolist()
            day_data.extend( self.stock_data.iloc[day_index-2].values.tolist() )
            day_data.extend([self.cash, self.position, mean(self.bought_stocks), self.last_sell])

            class_output, amount = strategy(day_data)

            current_price = self.stock_data.iloc[day_index].close
            if class_output == Action.SELL:
                shares_to_sell = floor(self.position * amount)
                profit_pct = self.sell_stock(current_price, shares_to_sell)
                if profit_pct >= 0:
                    self.actions.append((day_index, "Sell", current_price, profit_pct, shares_to_sell))
            elif class_output == Action.BUY:
                total_possible = floor(self.cash / current_price)
                shares_to_buy = floor(total_possible * amount)
                bought_price = self.buy_stock(current_price, shares_to_buy)
                if bought_price >= 0:
                    self.actions.append((day_index, "Buy", current_price, shares_to_buy))
            self.current_stock_value = self.position * current_price

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
