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

import collections

MAX_EPISODE_LENGTH = 100

class Episode(object):
    """ Sequence of actions and observations that correspond to a learning episode
    """

    def __init__(self):
        self.states = collections.deque([], MAX_EPISODE_LENGTH)
        self.values = collections.deque([], MAX_EPISODE_LENGTH)
        self.actions = collections.deque([], MAX_EPISODE_LENGTH)
        self.rewards = collections.deque([], MAX_EPISODE_LENGTH)
        self.cumulative_reward = 0.0

    def addState(self, state):
        """ Add a state observation to the episode
        """
        self.states.append(state)

    def addAction(self, action):
        """ Add an action to the episode. This allows the agent to keep track of
            its past actions.
        """
        self.actions.append(action)

    def addReward(self, reward):
        """ Add a reward to the episode. This is an instantaneous reward received
            after having performed an action.
        """
        self.rewards.append(reward)
        self.cumulative_reward += reward

    def addValues(self, values):
        """ Add values for actions of the last state.
        """
        self.values.append(values)
