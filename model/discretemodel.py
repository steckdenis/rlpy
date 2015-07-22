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

from .abstractmodel import *

class DiscreteModel(AbstractModel):
    """ Model used for storing values associated with discrete states, with no
        function approximation.
    """

    def __init__(self, nb_actions):
        super(DiscreteModel, self).__init__(nb_actions)

        self._data = {}

    def values(self, episode):
        """ Return the values of the last state of an episode
        """
        state = episode.states[-1]

        return [self._data.get(self.key(state, action), 0.0) for action in range(self.nb_actions)]

    def learn(self, episodes):
        """ Update the model using the updated values in the episodes.
        """
        for episode in episodes:
            for state, action, values in zip(episode.states, episode.actions, episode.values):
                self._data[self.key(state, action)] = values[action]

    def key(self, state, action):
        return tuple(state + (action,))