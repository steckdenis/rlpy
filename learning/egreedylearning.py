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

import random

from .abstractlearning import *

class EGreedyLearning(AbstractLearning):
    """ E-Greedy action selection
    """

    def __init__(self, nb_actions, learning, epsilon):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
            @param learning Learning method used when exploitation steps are taken
            @param epsilon Probability that an exploratory step is taken
        """
        super(EGreedyLearning, self).__init__(nb_actions)

        self.learning = learning
        self.epsilon = epsilon

    def actions(self, episode):
        # The best action has a probability 1-epsilon to be taken, the others share
        # a probability of epsilon
        actions, error = self.learning.actions(episode)

        best_index = 0
        best_proba = -1000000.0

        for index, a in enumerate(actions):
            if a > best_proba:
                best_proba = a
                best_index = index

        actions = [self.epsilon / (self.nb_actions - 1) for a in actions]
        actions[best_index] = 1.0 - self.epsilon

        return actions, error

    def finishEpisode(self, episode):
        self.learning.finishEpisode(episode)
