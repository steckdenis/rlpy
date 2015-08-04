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

class BatchAdvantageLearning(AbstractLearning):
    """ Advantage learning strategy, with the A-values updated at once when an
        episode is finished.
    """

    def __init__(self, nb_actions, alpha, gamma, kappa):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
            @param alpha Learning factor
            @param gamma Discount factor
            @param kappa The smaller this factor is, the strongest the bias towards
                         better-than-expected actions is.
        """
        super(BatchAdvantageLearning, self).__init__(nb_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa

    def actions(self, episode):
        """ Return the scores of the actions.

            @warning This is not a probability distribution (scores can be negative),
                     use SoftmaxLearning or EgreedyLearning in order to get
                     a real probability distribution !
        """
        return episode.values[-1], 0.0      # NOTE: The TD-error is not known yet because every computation is done in finishEpisode()

    def finishEpisode(self, episode):
        # Update the Advantage values from the end of the episode to the beginning
        for t in range(len(episode.states) - 1, 0, -1):
            prev_action = episode.actions[t - 1]
            prev_reward = episode.rewards[t - 1]
            prev_values = episode.values[t - 1]
            next_values = episode.values[t]

            advantage = prev_values[prev_action]
            value = max(prev_values)
            next_value = max(next_values)

            error = value + \
                    (prev_reward + self.gamma * next_value - value) / self.kappa - \
                    advantage

            prev_values[prev_action] = advantage + self.alpha * error

            if prev_reward > 9.0:
                print('big reward seen')
