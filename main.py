#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt

from gridworld import *
from polargridworld import *
from qlearning import *
from egreedylearning import *
from discretemodel import *

if __name__ == '__main__':

    if 'gridworld' in sys.argv:
        c = PolarGridWorld if 'polar' in sys.argv else GridWorld

        world = c(10, 5, (0, 2), (9, 2), (5, 2), 'stochastic' in sys.argv)

    if 'discrete' in sys.argv:
        model = DiscreteModel()

    if 'qlearning' in sys.argv:
        learning = QLearning(world.nb_actions(), model, 0.2, 0.9)

    if 'egreedy' in sys.argv:
        learning = EGreedyLearning(world.nb_actions(), learning, 0.1)

    # Perform simulation steps
    rewards = [0.0] * 2000

    for i in range(8):
        print('Iteration %i' % i)

        for it in range(2000):
            steps = 0

            state = world.initial
            last_reward = 0
            cumulative_reward = 0
            finished = False
            steps = 0

            world.reset()

            while steps < 2000 and not finished:
                action = learning.action(state, last_reward)
                state, last_reward, finished = world.performAction(action)

                cumulative_reward += last_reward
                steps += 1

            rewards[it] += cumulative_reward

    # Plot the results
    plt.plot([r / 8.0 for r in rewards])
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative reward')
    plt.savefig('rewards.pdf')
