import os
from functools import partial
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import neat

from config import *
from simulator import StockSimulator
from strategy import neat_strategy, buy_and_hold, moving_avg, rsi

simulator = StockSimulator().load_data(stock_ticker)

def visualizeActions(title, starting_index, sim_length, stock_data, actions, profit, avgLoss):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(title)
    sim_data = stock_data.iloc[starting_index:starting_index+sim_length]
    sim_data.close.plot(ax=ax1)
    sim_data.ema.plot(ax=ax1)
    sim_data.sma.plot(ax=ax1)
    ax1.title.set_text(f'Profit {profit:.2f}% AvgLoss {avgLoss:.2f}%')
    for action in actions:
        fc = 'r'
        if action[3] == 0 or action[3] > 1:
            fc = 'g'
        ax1.annotate(f'{action[1]} {action[3]:.2f}', (stock_data.iloc[action[0]].name, stock_data.iloc[action[0]].close), color=fc)
    sim_data.rsi.plot(ax=ax2)
    sim_data.obv.plot(ax=ax3)

def test():
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints/{}'.format(starting_checkpoint))
    if os.path.exists(checkpoint_path):
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        raise ValueError

    starting_index = 0#len(simulator.stock_data)-num_of_days_to_sim
    sim_length = len(simulator.stock_data)-starting_index
    net = neat.nn.FeedForwardNetwork.create(p.best_genome, p.config)
    genome_strategy = partial(neat_strategy, net)
    

    strategies = [
        ("NEAT", genome_strategy),
        ("Hold", buy_and_hold),
        ("Moving Avg", moving_avg),
        ("RSI", rsi),
    ]
    for name, strategy in strategies:
        simulator.reset_sim()
        simulator.sim_strategy(strategy, starting_index, sim_length)
        profit, avg_loss = simulator.evaluate()
        visualizeActions(name, starting_index, sim_length, simulator.stock_data, simulator.actions, profit, avg_loss)

    plt.show()

if __name__ == '__main__':
    test()