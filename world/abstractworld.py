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

import matplotlib.pyplot as plt
import math

from numpy.random import choice
from numpy import arange, ndarray

from .episode import *

def encode_identity(state):
    """ Identity encoding, does not change the state
    """
    return state

def _encode_onehot(state, ranges):
    """ Encode discrete state variables to a sequence of floats that each have
        a value of 0 or 1. For instance, (2, 3) is encoded as (0, 0, 1, 0, ...,
        0, 0, 0, 1, 0, ...).

        @param state State to encode
        @param ranges Ranges of the variables in the state (width and height of
                      a grid for instance)
    """
    res = [0.0] * sum(ranges)
    offset = 0

    for index, value in enumerate(state):
        res[offset + int(value + 0.49)] = 1.0
        offset += ranges[index]

    return tuple(res)

def make_encode_onehot(ranges):
    """ Return an encode function that encodes states using the provided ranges
    """
    return lambda s: _encode_onehot(s, ranges)

class AbstractWorld(object):
    """ Abstract world that receives actions and produces new states
    """
    def __init__(self):
        self.encoding = encode_identity

        self._min_state = [1e20] * 1000         # This big vector will be truncated the first time a state is encountered un run()
        self._max_state = [-1e20] * 1000

    def nb_actions(self):
        """ Return the number of actions that can be performed. This number
            cannot change during the lifetime of the world.
        """
        raise NotImplementedError('The world does not implement nb_actions()')

    def reset(self):
        """ Reset the world in its original configuration, as if no agent performed
            actions on it.
        """
        raise NotImplementedError('The world does not implement reset()')

    def performAction(self, action):
        """ Perform an action on the world and return a tuple (state, reward, finished)

            @note The state returned must not be encoded. AbstractWorld takes care
                  of the encoding when needed.
        """
        raise NotImplementedError('The world does not implement performAction()')

    def plotModel(self, model):
        """ Product PDF files that show graphically the values of a model that
            is used to represent this world. This function does not know the meaning
            of the values stored by the model.
        """
        episode = Episode()
        print('Plotting model')

        if len(self._min_state) == 1 or self._min_state[1] == self._max_state[1]:
            # 1D world
            X = []
            V = [[] for i in range(self.nb_actions())]

            mi = self._min_state[0]
            ma = self._max_state[0]

            for x in arange(mi, ma, (ma - mi) / 1000.0):
                X.append(x)

                episode.states.clear()
                episode.addState(self.encoding((x,)))

                values = model.values(episode)

                for action, value in enumerate(values):
                    V[action].append(value)

            # Plot a line graph
            for a in range(self.nb_actions()):
                plt.figure()

                plt.plot(X, V[a])
                plt.savefig('model_%i.pdf' % a)
        elif len(self._min_state) == 2:
            # 2D world
            miX = self._min_state[0]
            maX = self._max_state[0]
            miY = self._min_state[1]
            maY = self._max_state[1]

            V = [ndarray(shape=(50, 50)) for i in range(self.nb_actions())]
            py = 0

            for y in arange(miY, maY, (maY - miY) / 50.0):
                px = 0

                for x in arange(miX, maX, (maX - miX) / 50.0):
                    # Dummy episode that allows to fetch one value from the model
                    episode.states.clear()
                    episode.addState(self.encoding((x, y)))

                    values = model.values(episode)

                    for action, value in enumerate(values):
                        V[action][py, px] = value

                    px += 1
                py += 1

            # Plot a scatter plot
            for a in range(self.nb_actions()):
                plt.figure()

                plt.pcolormesh(V[a])
                plt.colorbar()
                plt.savefig('model_%i.pdf' % a)
        else:
            print('Unable to plot models of dimension 3 or above')

    def run(self, model, learning, num_episodes, max_episode_length, batch_size):
        """ Simulate an agent in this world.

            @param learning Learning algorithm used by the agent
            @param model Model that will be used for learning
            @param num_episodes Number of episodes that are simulated
            @param max_episode_length Maximum number of steps allowed per episode
            @param batch_size Off-policy learning happens once after every @p batch_size episodes

            @return A list of Episode objects
        """
        episodes = []
        learn_episodes = []
        possible_actions = list(range(self.nb_actions()))

        try:
            for e in range(num_episodes):
                episode = Episode()

                # Initial state
                self.reset()

                episode.addState(self.encoding(self.initial))
                episode.addValues(model.values(episode))

                finished = False
                steps = 0

                # Perform the steps
                while steps < max_episode_length and not finished:
                    probas, _ = learning.actions(episode)
                    action = choice(possible_actions, p=probas)

                    state, reward, finished = self.performAction(action)

                    self._min_state = [min(a, b) for a, b in zip(self._min_state, state)]
                    self._max_state = [max(a, b) for a, b in zip(self._max_state, state)]

                    episode.addReward(reward)
                    episode.addAction(action)
                    episode.addState(self.encoding(state))
                    episode.addValues(model.values(episode))

                    steps += 1

                # Let the learning update the Q-value of the last state visited
                learning.actions(episode)

                # If a batch has been finished, learn
                episodes.append(episode)
                learn_episodes.append(episode)

                print(e, episode.cumulative_reward)

                if len(learn_episodes) == batch_size:
                    model.learn(learn_episodes)
                    learn_episodes = []
        except KeyboardInterrupt:
            # Allow the user to gracefully interrupt the learning process
            pass

        return episodes
