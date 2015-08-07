#!/usr/bin/python3
#
# Copyright (c) 2015 Vrije Universiteit Brussel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
import matplotlib.pyplot as plt

from world.gridworld import *
from world.polargridworld import *
from world.pogridworld import *
from world.tmazeworld import *
from world.rlglueworld import *
from world.rosworld import *
from learning.qlearning import *
from learning.batchqlearning import *
from learning.advantagelearning import *
from learning.batchadvantagelearning import *
from learning.egreedylearning import *
from learning.softmaxlearning import *
from learning.adaptivesoftmaxlearning import *
from model.discretemodel import *
from model.grumodel import *
from model.mut1model import *
from model.mut2model import *
from model.mut3model import *
from model.lstmmodel import *
from model.clstmmodel import *
from model.kerasnnetmodel import *
from model.fannnnetmodel import *
from texplore.texploremodel import *

try:
    import std_msgs
except ImportError:
    pass

try:
    import theano

    theano.config.allow_gc = False
    theano.config.linker = 'cvm_nogc'
    theano.config.openmp = True
except ImportError:
    print('Theano not installed, several nnet-based models will not be usable')

EPISODES = 5000
MAX_TIMESTEPS = 500
BATCH_SIZE = 10
DISCOUNT_FACTOR = 0.90

HISTORY_LENGTH = 10
HIDDEN_NEURONS = 100
SOFTMAX_TEMP = 0.5

if __name__ == '__main__':

    if 'gridworld' in sys.argv:
        world = GridWorld(10, 5, (0, 2), (9, 2), (5, 2), 'stochastic' in sys.argv)
    elif 'pogridworld' in sys.argv:
        world = POGridWorld(10, 5, (0, 2), (9, 2), (5, 2), 'stochastic' in sys.argv)
    elif 'polargridworld' in sys.argv:
        world = PolarGridWorld(10, 5, (0, 2), (9, 2), (5, 2), 'stochastic' in sys.argv)
    elif 'tmaze' in sys.argv:
        EPISODES = 50000

        world = TMazeWorld(8, 1)
    elif 'rlglue' in sys.argv:
        # Let the RL-Glue experiment orchestrate everything
        MAX_TIMESTEPS = 1000000000
        EPISODES = 1000000000
        world = RLGlueWorld()
    elif 'rospendulum' in sys.argv:
        # Toy ROS experiment : inverted pendulum. The agent senses the angle
        # and angular velocity of the pendulum, and can apply force on it.
        MAX_TIMESTEPS = 1000
        BATCH_SIZE = 1
        DISOUNT_FACTOR = 0.95

        subscriptions = [
            {'path': '/vrep/jointAngle', 'type': std_msgs.msg.Float32},
            {'path': '/vrep/jointVelocity', 'type': std_msgs.msg.Float32},
            {'path': '/vrep/reward', 'type': std_msgs.msg.Float32},
        ]
        publications = [
            {'path': '/vrep/jointTorque', 'type': std_msgs.msg.Float64, 'values': [-1.0, 0.0, 1.0]},
            {'path': '/vrep/reset', 'type': std_msgs.msg.Int32, 'values': [1]},
        ]

        world = ROSWorld(subscriptions, publications)

    if 'discrete' in sys.argv:
        makemodel = lambda n: DiscreteModel(n)
    elif 'gru' in sys.argv:
        makemodel = lambda n: GRUModel(n, HISTORY_LENGTH, HIDDEN_NEURONS)
    elif 'mut1' in sys.argv:
        makemodel = lambda n: MUT1Model(n, HISTORY_LENGTH, HIDDEN_NEURONS)
    elif 'mut2' in sys.argv:
        makemodel = lambda n: MUT2Model(n, HISTORY_LENGTH, HIDDEN_NEURONS)
    elif 'mut3' in sys.argv:
        makemodel = lambda n: MUT3Model(n, HISTORY_LENGTH, HIDDEN_NEURONS)
    elif 'lstm' in sys.argv:
        makemodel = lambda n: LSTMModel(n, HISTORY_LENGTH, HIDDEN_NEURONS)
    elif 'clstm' in sys.argv:
        makemodel = lambda n: CLSTMModel(n, HIDDEN_NEURONS)
    elif 'kerasnnet' in sys.argv:
        makemodel = lambda n: KerasNnetModel(n, HIDDEN_NEURONS)
    elif 'fannnnet' in sys.argv:
        makemodel = lambda n: FannNnetModel(n, HIDDEN_NEURONS)

    model = makemodel(world.nb_actions())

    if 'oneofn' in sys.argv:
        world.encoding = make_encode_onehot([10, 5])

    if 'qlearning' in sys.argv:
        learning = QLearning(world.nb_actions(), 0.2, DISCOUNT_FACTOR)
    elif 'batchqlearning' in sys.argv:
        learning = BatchQLearning(world.nb_actions(), 0.6, DISCOUNT_FACTOR)
    elif 'advantage' in sys.argv:
        learning = AdvantageLearning(world.nb_actions(), 0.2, DISCOUNT_FACTOR, 0.3)
    elif 'batchadvantage' in sys.argv:
        learning = BatchAdvantageLearning(world.nb_actions(), 0.6, DISCOUNT_FACTOR, 0.3)

    if 'egreedy' in sys.argv:
        learning = EGreedyLearning(world.nb_actions(), learning, 0.1)
    elif 'softmax' in sys.argv:
        learning = SoftmaxLearning(world.nb_actions(), learning, SOFTMAX_TEMP)
    elif 'adaptivesoftmax' in sys.argv:
        learning = AdaptiveSoftmaxLearning(world.nb_actions(), learning, HIDDEN_NEURONS, 0.1)

    if 'texplore' in sys.argv:
        BATCH_SIZE = 1

        model = TExploreModel(
            world,
            makemodel,
            model,
            learning,
            50
        )

    # Perform simulation steps
    episodes = world.run(model, learning, EPISODES, MAX_TIMESTEPS, BATCH_SIZE)

    # Plot the cumulative reward of all the episodes
    plt.figure()
    plt.plot([e.cumulative_reward for e in episodes])
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative reward')
    plt.savefig('rewards.pdf')

    # Plot the model
    world.plotModel(model)
