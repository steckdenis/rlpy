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
from oneofnmodel import *

import theano

theano.config.allow_gc = False
theano.config.linker = 'cvm_nogc'
theano.config.openmp = True

ITERATIONS = 1
EPISODES = 1000
MAX_TIMESTEPS = 5000

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
        model = OneOfNModel(world.nb_actions(), model, [10, 5])

    if 'qlearning' in sys.argv:
        learning = QLearning(world.nb_actions(), model, 0.2, 0.9)

    if 'egreedy' in sys.argv:
        learning = EGreedyLearning(world.nb_actions(), learning, 0.1)

    # Perform simulation steps
    rewards = [0.0] * EPISODES

    for i in range(ITERATIONS):
        print('Iteration %i' % i)

        for episode in range(EPISODES):
            steps = 0

            state = world.initial
            last_reward = 0
            cumulative_reward = 0
            finished = False
            steps = 0

            world.reset()
            model.nextEpisode()

            while steps < MAX_TIMESTEPS and not finished:
                action = learning.action(state, last_reward)
                state, last_reward, finished = world.performAction(action)

                cumulative_reward += last_reward
                steps += 1

            rewards[episode] += cumulative_reward
            print(cumulative_reward)

    # Plot the results
    plt.plot([r / ITERATIONS for r in rewards])
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative reward')
    plt.savefig('rewards.pdf')
