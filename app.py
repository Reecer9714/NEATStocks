
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

stock_inputs = []
stock_outputs = []
indexes = []
stock_data = []
num_of_data_points = 100
starting_money = 5000
num_of_days_to_sim = 12
num_of_days_to_lookback = 7


def load_data(symbol):
    data = pd.read_pickle(os.path.join(local_dir, f'data/{symbol}-intraday'))
    df = pd.DataFrame(data)

    stock_data = df
    # for index in indexes:
    #     stock_inputs.append(
    #         df.loc[index, ['1. open', '2. high', '3. low', '5. adjusted close']])
    #     stock_outputs.append(df.loc[index-1, ['5. adjusted close']])

    return indexes, stock_data


def calc_fitness(genome, config):
    indexes = random.sample(range(
        num_of_days_to_lookback*24, len(stock_data)-num_of_days_to_sim*24), num_of_data_points)
    fitness = 1
    money_sum = 0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for starting_index in indexes:
        money = starting_money
        position = 0
        last_buy = 0
        last_sell = 0
        made_profit = False
        profit_pct = 0
        tried_buy = False
        tried_sell = False
        for day in range(num_of_days_to_sim):
            day_index = starting_index + day * 24
            day_data = stock_data.loc[day_index,
                                      ['1. open', '2. high', '3. low', '4. close']].values.tolist()
            day_data.append(money)
            day_data.append(position)
            net_output = net.activate(day_data)
            softmax_result = softmax(net_output)
            class_output = np.argmax(
                ((softmax_result / np.max(softmax_result)) == 1).astype(int))
            if class_output == 1 and position > 5:
                # Sell
                position -= 5
                last_sell = day_data[0]
                money += day_data[0] * 5
                tried_sell = True
                profit_pct = last_sell / last_buy
                if last_sell > last_buy:
                    made_profit = True
                fitness *= profit_pct
            elif class_output == 2 and day_data[3] * 5 < money:
                # Buy
                position = position + 5
                last_buy = day_data[3]
                money -= day_data[3] * 5
                tried_buy = True
            if made_profit:
                fitness *= 1.05
        fitness *= money / starting_money
        # fitness = fitness + money
    # fitness = fitness / len(indexes)
    if not tried_buy:
        fitness = 0
    else:
        fitness *= 1.01
    # if tried_sell:
    #     fitness = fitness * 1.1
    # fitness = fitness / (starting_money * len(indexes))
    return fitness


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p = neat.Checkpointer.restore_checkpoint('checkpoints/stock-checkpoint-177')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(
        50, filename_prefix="checkpoints/stock-checkpoint-"))

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), calc_fitness)
    winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

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
    indexes, stock_data = load_data('AAPL')
    run(config_path)
