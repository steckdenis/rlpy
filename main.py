#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt

from gridworld import *
from polargridworld import *
from qlearning import *
from egreedylearning import *
from discretemodel import *
from lstmmodel import *
from nnetmodel import *
from oneofnworld import *

import theano

theano.config.allow_gc = False
theano.config.linker = 'cvm_nogc'
theano.config.openmp = True

EPISODES = 1000
MAX_TIMESTEPS = 5000
BATCH_SIZE = 10

if __name__ == '__main__':

    if 'gridworld' in sys.argv:
        c = PolarGridWorld if 'polar' in sys.argv else GridWorld

        world = c(10, 5, (0, 2), (9, 2), (5, 2), 'stochastic' in sys.argv)

    if 'discrete' in sys.argv:
        model = DiscreteModel(world.nb_actions())
    elif 'lstm' in sys.argv:
        model = LSTMModel(world.nb_actions())
    elif 'nnet' in sys.argv:
        model = NnetModel(world.nb_actions(), 200)

    if 'oneofn' in sys.argv:
        world = OneOfNWorld(world, [10, 5])

    if 'qlearning' in sys.argv:
        learning = QLearning(world.nb_actions(), 0.2, 0.8)

    if 'egreedy' in sys.argv:
        learning = EGreedyLearning(world.nb_actions(), learning, 0.1)

    # Perform simulation steps
    episodes = world.run(model, learning, EPISODES, MAX_TIMESTEPS, BATCH_SIZE)

    world.plotModel(model)

    # Plot the cumulative reward of all the episodes
    plt.figure()
    plt.plot([sum(e.rewards) for e in episodes])
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative reward')
    plt.savefig('rewards.pdf')
