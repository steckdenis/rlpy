#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt

from world.gridworld import *
from world.polargridworld import *
from world.rlglueworld import *
from learning.qlearning import *
from learning.advantagelearning import *
from learning.egreedylearning import *
from learning.softmaxlearning import *
from model.discretemodel import *
from model.lstmmodel import *
from model.clstmmodel import *
from model.kerasnnetmodel import *
from model.fannnnetmodel import *

try:
    import theano

    theano.config.allow_gc = False
    theano.config.linker = 'cvm_nogc'
    theano.config.openmp = True
except ImportError:
    print('Theano not installed, several nnet-based models will not be usable')

EPISODES = 1000
MAX_TIMESTEPS = 5000
BATCH_SIZE = 10

if __name__ == '__main__':

    if 'gridworld' in sys.argv:
        c = PolarGridWorld if 'polar' in sys.argv else GridWorld

        world = c(10, 5, (0, 2), (9, 2), (5, 2), 'stochastic' in sys.argv)
    elif 'rlglue' in sys.argv:
        # Let the RL-Glue experiment orchestrate everything
        MAX_TIMESTEPS = 1000000000
        EPISODES = 1000000000
        world = RLGlueWorld()

    if 'discrete' in sys.argv:
        model = DiscreteModel(world.nb_actions())
    elif 'lstm' in sys.argv:
        model = LSTMModel(world.nb_actions(), 20, 100)
    elif 'clstm' in sys.argv:
        model = CLSTMModel(world.nb_actions(), 100)
    elif 'kerasnnet' in sys.argv:
        model = KerasNnetModel(world.nb_actions(), 200)
    elif 'fannnnet' in sys.argv:
        model = FannNnetModel(world.nb_actions(), 200)

    if 'oneofn' in sys.argv:
        world.encoding = make_encode_onehot([10, 5])

    if 'qlearning' in sys.argv:
        learning = QLearning(world.nb_actions(), 0.2, 0.8)
    elif 'advantage' in sys.argv:
        learning = AdvantageLearning(world.nb_actions(), 0.2, 0.8, 0.3)

    if 'egreedy' in sys.argv:
        learning = EGreedyLearning(world.nb_actions(), learning, 0.1)
    elif 'softmax' in sys.argv:
        learning = SoftmaxLearning(world.nb_actions(), learning, 0.2)

    # Perform simulation steps
    episodes = world.run(model, learning, EPISODES, MAX_TIMESTEPS, BATCH_SIZE)

    # Plot the cumulative reward of all the episodes
    plt.figure()
    plt.plot([sum(e.rewards) for e in episodes])
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative reward')
    plt.savefig('rewards.pdf')

    # Plot the model
    world.plotModel(model)
