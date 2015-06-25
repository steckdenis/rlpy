#!/usr/bin/python3

from gridworld import *
from qlearning import *
from egreedylearning import *
from discretemodel import *

if __name__ == '__main__':
    world = GridWorld(10, 5, (0, 2), (9, 2), (5, 2))
    model = DiscreteModel()
    learning = QLearning(world.nb_actions(), model, 0.2, 0.9)
    learning = EGreedyLearning(world.nb_actions(), learning, 0.05)

    # Perform simulation steps
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

        print('%i;%i' % (it, cumulative_reward))