import os
from functools import partial
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import neat

from config import *
import visualize
from simulator import StockSimulator
from strategy import *

simulator = StockSimulator().load_data(stock_ticker)

def plot_Value(axis, cash, value, capital, timeline):
    axis.stackplot(timeline, cash, value, colors=['g','b'])
    axis.plot(timeline, capital, color='r')
    axis.title.set_text('Value')

def plot_RSI(axis, rsi):
    rsi.plot(ax=axis, color='b')
    axis.axhline(y=30)
    axis.axhline(y=70)
    axis.title.set_text('RSI')

def plot_STOCH(axis, slowd, slowk):
    slowd.plot(ax=axis, color='r')
    slowk.plot(ax=axis, color='g')
    axis.axhline(y=20)
    axis.axhline(y=80)
    axis.title.set_text('STOCH')

def visualizeActions(strategy_name, starting_index, sim_length, simulator: StockSimulator, profit, stats):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    fig.suptitle(f'Profit {profit:.2f}% AvgLoss {stats["losses"]["mean"]:.2f}%+-{stats["losses"]["stdev"]:.2f}')
    simulator.visualize(ax1, starting_index, sim_length, strategy_name)
    sim_data = simulator.stock_data.iloc[starting_index:starting_index+sim_length]
    plot_Value(ax2, simulator.cash_history, simulator.value_history, simulator.capital_history, sim_data.axes[0])
    plot_RSI(ax3, sim_data.rsi)
    plot_STOCH(ax4, sim_data.slowd, sim_data.slowk)

def test():
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints/{}'.format(starting_checkpoint))
    if os.path.exists(checkpoint_path):
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:
        raise FileNotFoundError(checkpoint_path)

    starting_index = 0#len(simulator.stock_data)-num_of_days_to_sim
    sim_length = len(simulator.stock_data)-starting_index
    net = neat.nn.FeedForwardNetwork.create(p.best_genome, p.config)
    genome_strategy = partial(neat_strategy, net)

    strategies = [
        ("NEAT", genome_strategy),
        ("Hold", buy_and_hold),
        # ("Moving Avg", moving_avg_check),
        # ("STOCH", stoch_check),
        # ("Buy Avg, Sell RSI", combo_ma_stoch),
        # ("STOCH Change", stochStrategy)
    ]
    for name, strategy in strategies:
        simulator.reset_sim()
        simulator.sim_strategy(strategy, starting_index, sim_length)
        profit, stats = simulator.evaluate()
        visualizeActions(name, starting_index, sim_length, simulator, profit, stats)
        print(name, stats)
        if name == "NEAT":
            visualize.draw_net(p.config, p.best_genome, view=True, node_names=node_names, show_disabled=False)

    plt.show()

if __name__ == '__main__':
    test()