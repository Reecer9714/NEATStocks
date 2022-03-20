import os
from functools import partial
from random import randint
import neat
import visualize
import multiprocessing

from config import *
from simulator import StockSimulator
from strategy import neat_strategy

simulator = StockSimulator().load_data(stock_ticker)

def calc_fitness(genome, config):
    starting_index = len(simulator.stock_data)-num_of_days_to_sim
    sim_length = len(simulator.stock_data)-starting_index
    # starting_index = randint(num_of_days_to_lookback,len(simulator.stock_data)-num_of_days_to_sim)
    simulator.reset_sim()
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    genome_strategy = partial(neat_strategy, net)
    simulator.sim_strategy(genome_strategy, starting_index, sim_length)
    fitness, _ = simulator.evaluate()
    
    return fitness

def run(config_file, reporter=neat.StdOutReporter(True)):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints/{}'.format(starting_checkpoint))
    if os.path.exists(checkpoint_path):
        p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    p.add_reporter(reporter)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(
        checkpoint_generations, filename_prefix='checkpoints/{}-checkpoint-'.format(stock_ticker)))

    # Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), calc_fitness)
    winner = p.run(pe.evaluate, num_of_generations)
    
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    genome_strategy = partial(neat_strategy, net)
    simulator.sim_strategy(genome_strategy, len(simulator.stock_data)-num_of_days_to_sim, num_of_days_to_sim)
    profit, avg_loss = simulator.evaluate()

    # Display the winning genome.
    print('\nBest genome Profit({}) AvgLoss({}):\n{}'.format(profit-1, avg_loss-1, winner))

    visualize.draw_net(config, winner, view=True, node_names=node_names, show_disabled=False)
    # stats_fig = visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    run(config_path)
