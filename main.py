#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt

from world.gridworld import *
from world.polargridworld import *
from world.oneofnworld import *
from learning.qlearning import *
from learning.advantagelearning import *
from learning.egreedylearning import *
from learning.softmaxlearning import *
from model.discretemodel import *
from model.lstmmodel import *
from model.nnetmodel import *

import theano

theano.config.allow_gc = False
theano.config.linker = 'cvm'
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
    elif 'advantage' in sys.argv:
        learning = AdvantageLearning(world.nb_actions(), 0.2, 0.8, 0.5)     # Kappa of 0.1 as used in "Reinforcement Learning with Long Short-Term Memory"

    if 'egreedy' in sys.argv:
        learning = EGreedyLearning(world.nb_actions(), learning, 0.1)
    elif 'softmax' in sys.argv:
        learning = SoftmaxLearning(world.nb_actions(), learning, 0.2)

    # Perform simulation steps
    episodes = world.run(model, learning, EPISODES, MAX_TIMESTEPS, BATCH_SIZE)

    world.plotModel(model)

    # Plot the cumulative reward of all the episodes
    plt.figure()
    plt.plot([sum(e.rewards) for e in episodes])
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative reward')
    plt.savefig('rewards.pdf')
