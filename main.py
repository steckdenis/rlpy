#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt

from gridworld import *
from qlearning import *
from egreedylearning import *
from discretemodel import *

if __name__ == '__main__':

    if 'gridworld' in sys.argv:
        world = GridWorld(10, 5, (0, 2), (9, 2), (5, 2))

    if 'discrete' in sys.argv:
        model = DiscreteModel()

    if 'qlearning' in sys.argv:
        learning = QLearning(world.nb_actions(), model, 0.2, 0.9)

    if 'egreedy' in sys.argv:
        learning = EGreedyLearning(world.nb_actions(), learning, 0.1)

    # Perform simulation steps
    rewards = []

    for it in range(5000):
        steps = 0

        state = world.initial
        last_reward = 0
        cumulative_reward = 0
        finished = False

        world.reset()

        while not finished:
            action = learning.action(state, last_reward)
            state, last_reward, finished = world.performAction(action)

            cumulative_reward += last_reward

        rewards.append(cumulative_reward)

    # Plot the results
    plt.plot(rewards)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative reward')
    plt.savefig('rewards.pdf')
