
import os
import neat
from neat.math_util import softmax
import pandas as pd
import numpy as np
import random
import visualize
import multiprocessing

stock_ticker = 'AAPL'
starting_checkpoint = 'AAPL-checkpoint-554'
stock_data = pd.DataFrame()
num_of_starting_points = 30
starting_money = 5000
num_of_days_to_sim = 5 * 8
num_of_days_to_lookback = 0
postitions_to_buysell = 5
num_of_generations = 500


def load_data(symbol):
    data = pd.read_pickle(os.path.join(local_dir, f'data/{symbol}-intraday'))

    return data


def sim_network(net, starting_index):
    cash = starting_money
    position = 0
    last_buy = 0
    last_sell = 0
    profit_pct = 0
    avg_sell_profit = 0
    avg_loss = 0
    num_sells = 0
    num_loss = 0
    tried_buy = False
    current_stock_value = 0
    for day in range(num_of_days_to_sim):
        day_index = starting_index + day
        day_data = stock_data.iloc[day_index].values.tolist()
        day_data.extend([cash, position, last_buy, last_sell])
        net_output = net.activate(day_data)
        softmax_result = softmax(net_output)
        class_output = np.argmax(
            ((softmax_result / np.max(softmax_result)) == 1).astype(int))
        close_price = day_data[3]
        if class_output == 1 and position > postitions_to_buysell:
            # Sell
            position -= postitions_to_buysell
            last_sell = close_price
            cash += close_price * postitions_to_buysell
            profit_pct = last_sell / last_buy
            avg_sell_profit += profit_pct
            num_sells += 1
            if profit_pct < 1:
                avg_loss += profit_pct
                num_loss += 1
        elif class_output == 2 and close_price * postitions_to_buysell < cash:
            # Buy
            position = position + postitions_to_buysell
            last_buy = close_price
            cash -= close_price * postitions_to_buysell
            tried_buy = True
        current_stock_value = position * close_price
    if num_sells > 0:
        avg_sell_profit /= num_sells
    if num_loss > 0:
        avg_loss /= num_loss
    else:
        avg_loss = 0.92 # Assume 8% loss
    overal_profit = (current_stock_value * avg_loss + cash) / starting_money
    if not tried_buy:
        overal_profit = 0
    return overal_profit, avg_sell_profit

def calc_fitness(genome, config):
    indexes = random.sample(
        range(num_of_days_to_lookback, len(stock_data)-num_of_days_to_sim),
        num_of_starting_points)
    indexes.append(len(stock_data)-num_of_days_to_sim)
    fitness = 0
    
    net = neat.nn.RecurrentNetwork.create(genome, config)
    for starting_index in indexes:
        net_fitness, _ = sim_network(net, starting_index)
        fitness += net_fitness
    
    return fitness / len(indexes)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    checkpoint_path = os.path.join(os.path.dirname(__file__), f'checkpoints/{starting_checkpoint}')
    if os.path.exists(checkpoint_path):
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(
        num_of_generations, filename_prefix=f'checkpoints/{stock_ticker}-checkpoint-'))

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), calc_fitness)
    winner = p.run(pe.evaluate, num_of_generations+1)
    
    net = neat.nn.RecurrentNetwork.create(winner, config)
    profit, avg_sell_profit = sim_network(net, len(stock_data)-num_of_days_to_sim)

    # Display the winning genome.
    print(f'\nBest genome Profit({profit}) ASP({avg_sell_profit}):\n{winner}')

    node_names = {
        -1: 'open', -2: 'high', -3: 'low', -4: 'close',
        -5: 'rsi', -6: 'ema', -7: 'sma', -8: 'obv',
        -9: 'cash', -10: 'pos', -11: 'last_buy', -12: 'last_sell',
        0: 'hold', 1: 'sell', 2: 'buy'}
    visualize.draw_net(config, winner, view=True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=False)
    # visualize.plot_species(stats, view=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'stock-feedforward')
    stock_data = load_data(stock_ticker)
    run(config_path)
