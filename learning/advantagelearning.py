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

class AdvantageLearning(AbstractLearning):
    """ Advantage Learning learning strategy
    """

    def __init__(self, nb_actions, alpha, gamma, kappa):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
            @param model Model used to store the Q-values. Can be any discrete or
                         continuous model. 
            @param alpha Learning factor
            @param gamma Discount factor
            @param kappa The smaller this factor is, the strongest the bias towards
                         better-than-expected actions is.
        """
        super(AdvantageLearning, self).__init__(nb_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa

    def actions(self, episode):
        """ Return the advantage values of the actions. See the warning in QLearning
            about the fact that those are not probabilities.
        """

        # Update the Advantage value of the last action that was taken
        if len(episode.actions) > 0:
            last_action = episode.actions[-1]
            last_reward = episode.rewards[-1]

            advantage = episode.values[-2][last_action]
            value = max(episode.values[-2])
            next_value = max(episode.values[-1])
            error = value + \
                    (last_reward + self.gamma * next_value - value) / self.kappa - \
                    advantage

            episode.values[-2][last_action] = advantage + self.alpha * error
        else:
            error = 0.0

        # Probability to take any of the actions
        return episode.values[-1], error
