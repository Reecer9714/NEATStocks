
from numpy import argmax
from neat.math_util import softmax
from simulator import Action

def neat_strategy(net, day_data):
    net_output = net.activate(day_data)
    softmax_result = softmax(net_output)
    class_output = Action(argmax(softmax_result))
    return class_output

def buy_and_hold(day_data):
    '''Buy as much as possible then hold'''
    return Action.BUY

def moving_avg(day_data):
    '''Buy when short period mvAvg is higher than long period'''
    if day_data[5] > day_data[6]:
        return Action.BUY
    return Action.SELL

def rsi(day_data):
    '''Buy when RSI is low, and sell when high'''
    if day_data[4] < 30:
        return Action.BUY
    elif day_data[4] > 70:
        return Action.SELL
    return Action.HOLD