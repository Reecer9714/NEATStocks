
from numpy import argmax
from neat.math_util import softmax
from simulator import Action

def neat_strategy(net, day_data):
    net_output = net.activate(day_data)
    softmax_result = softmax(net_output)
    class_output = Action(argmax(softmax_result))
    return class_output