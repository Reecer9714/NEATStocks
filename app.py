
import os
from itertools import repeat
from functools import partial
import neat
from neat.math_util import softmax
import pandas as pd
import numpy as np
import random
import visualize
import multiprocessing

stock_ticker = 'AAPL'
indexes = []
stock_data = []
num_of_data_points = 50
starting_money = 5000
num_of_days_to_sim = 14
num_of_days_to_lookback = 7
postitions_to_buysell = 5


def load_data(symbol):
    data = pd.read_pickle(os.path.join(local_dir, f'data/{symbol}-intraday'))
    df = pd.DataFrame(data)

    return df


def sim_network(net, starting_index):
    money = starting_money
    position = 0
    last_buy = 0
    last_sell = 0
    profit_pct = 0
    week_fitness = 1
    tried_buy = False
    for day in range(num_of_days_to_sim):
        made_profit = False
        day_index = starting_index + day * 24
        day_data = stock_data.loc[day_index,
                                    ['1. open', '2. high', '3. low', '4. close']].values.tolist()
        day_data.append(money)
        day_data.append(position)
        net_output = net.activate(day_data)
        softmax_result = softmax(net_output)
        class_output = np.argmax(
            ((softmax_result / np.max(softmax_result)) == 1).astype(int))
        if class_output == 1 and position > postitions_to_buysell:
            # Sell
            position -= postitions_to_buysell
            last_sell = day_data[0]
            money += day_data[0] * postitions_to_buysell
            profit_pct = last_sell / last_buy
            week_fitness *= profit_pct
        elif class_output == 2 and day_data[3] * postitions_to_buysell < money:
            # Buy
            position = position + postitions_to_buysell
            last_buy = day_data[3]
            money -= day_data[3] * postitions_to_buysell
            tried_buy = True
    profit = money / starting_money
    net_fitness = week_fitness * profit
    if not tried_buy:
        net_fitness = 0
    return net_fitness, profit

def calc_fitness(genome, config):
    indexes = random.sample(range(
        num_of_days_to_lookback*24, len(stock_data)-num_of_days_to_sim*24), num_of_data_points)
    fitness = 0
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for starting_index in indexes:
        net_fitness, _ = sim_network(net, starting_index)
        fitness += net_fitness
    
    return fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p = neat.Checkpointer.restore_checkpoint('checkpoints/stock-checkpoint-78')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(
        10, filename_prefix="checkpoints/stock-checkpoint-"))

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), calc_fitness)
    winner = p.run(pe.evaluate, 30)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    _, profit = sim_network(net, len(stock_data)-num_of_days_to_sim*24)

    # Display the winning genome.
    print(f'\nBest genome Profit({profit}):\n{winner}')

    node_names = {-1: 'open', -2: 'high', -
                  3: 'low', -4: 'close', -5: 'money', -6: 'pos', 0: 'hold', 1: 'sell', 2: 'buy'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'stock-feedforward')
    stock_data = load_data(stock_ticker)
    run(config_path)
