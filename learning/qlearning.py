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

from .abstractlearning import *

class QLearning(AbstractLearning):
    """ Q-Learning learning strategy
    """

    def __init__(self, nb_actions, alpha, gamma):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
            @param model Model used to store the Q-values. Can be any discrete or
                         continuous model. 
            @param alpha Learning factor
            @param gamma Discount factor
        """
        super(QLearning, self).__init__(nb_actions)

        self.alpha = alpha
        self.gamma = gamma

    def actions(self, episode):
        """ Return the scores of the actions.

            @warning This is not a probability distribution (scores can be negative),
                     use SoftmaxLearning or EgreedyLearning in order to get
                     a real probability distribution !
        """

        # Update the Q-value of the last action that was taken
        if len(episode.actions) > 0:
            last_action = episode.actions[-1]
            last_reward = episode.rewards[-1]

            Q = episode.values[-2][last_action]
            error = last_reward + self.gamma * max(episode.values[-1]) - Q

            episode.values[-2][last_action] = Q + self.alpha * error

        # Values of the actions
        return episode.values[-1]