
from numpy import argmax, max, sign
from neat.math_util import softmax
from simulator import Action
from config import Inputs

def line_intersect(line1, line2):
    '''Returns the direction of the intersection or 0 if not intersecting'''
    if line1[0] > line2[0] and line1[1] < line2[1]:
        return -1
    elif line1[0] < line2[0] and line1[1] > line2[1]:
        return 1
    return 0
    
def neat_strategy(net, day_data):
    net_output = net.activate(day_data)
    try:
        softmax_result = softmax(net_output[:3])
        class_output = Action(argmax(softmax_result))
    except OverflowError as err:
        print(f"OverflowError: {err} {net_output}")
        return Action.HOLD, 0
    return class_output, net_output[3]

def buy_and_hold(day_data):
    '''Buy as much as possible then hold'''
    return Action.BUY, 1

def moving_avg_check(day_data):
    if day_data[Inputs.EMA.value] > day_data[Inputs.SMA.value]:
        return Action.BUY, 1
    return Action.SELL, 1

def moving_avg_change(day_data):
    '''Buy when short period mvAvg is higher than long period'''
    direction = line_intersect(
        (day_data[Inputs.LAST_EMA.value], day_data[Inputs.EMA.value]),
        (day_data[Inputs.LAST_SMA.value], day_data[Inputs.SMA.value])
    )
    if direction > 0:
        return Action.BUY, 1
    elif direction < 0:
        return Action.SELL, 1
    return Action.HOLD, 1

def rsi_check(day_data):
    '''Buy when RSI is low, and sell when high'''
    if day_data[Inputs.RSI.value] < 30:
        return Action.BUY, 1
    elif day_data[Inputs.RSI.value] > 70:
        return Action.SELL, 1
    return Action.HOLD, 1

def rsi_change(day_data):
    if day_data[Inputs.RSI.value] > 30 and day_data[Inputs.LAST_RSI.value] < 30:
        return Action.BUY, 1
    elif day_data[Inputs.RSI.value] < 70 and day_data[Inputs.LAST_RSI.value] < 70:
        return Action.SELL, 1
    return Action.HOLD, 1

def stoch_check(day_data):
    '''Buy when STOCK is low, and sell when high'''
    if day_data[Inputs.SLOWK.value] < 20:
        return Action.BUY, 1
    elif day_data[Inputs.SLOWK.value] > 80:
        return Action.SELL, 1
    return Action.HOLD, 1

def stoch_change(day_data):
    '''Buy when slowk crosses slowd positively, Sell when slowk crosses slowd negatively'''
    direction = line_intersect(
        (day_data[Inputs.LAST_SLOWK.value], day_data[Inputs.SLOWK.value]),
        (day_data[Inputs.LAST_SLOWD.value], day_data[Inputs.SLOWD.value])
    )
    if direction > 0:
        return Action.SELL, 1
    elif direction < 0:
        return Action.BUY, 1
    return Action.HOLD, 1

def stochStrategy(day_data):
    stochAction = stoch_check(day_data)
    if stochAction == stoch_change(day_data):
        return stochAction
    return Action.HOLD, 1

def combo_ma_stoch(day_data):
    '''Buy based on moving_avg, Sell based on rsi'''
    rsiAction = stoch_change(day_data)
    if rsiAction == Action.SELL and moving_avg_change(day_data) == Action.SELL:
        return Action.SELL, 1
    elif rsiAction == Action.BUY and moving_avg_change(day_data) == Action.BUY:
        return Action.BUY, 1
    return Action.HOLD, 1