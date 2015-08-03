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

import math

from .abstractlearning import *

def _bounded_exp(x):
    """ Exponential with clamping of big and small X values
    """
    if x < -699:
        return 1e-305
    elif x > 699:
        return 1e304
    else:
        return math.exp(x)

class SoftmaxLearning(AbstractLearning):
    """ Softmax action selection
    """

    def __init__(self, nb_actions, learning, temperature):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
            @param learning Learning method used when exploitation steps are taken
            @param temperature Temperature used to balance exploration/exploitation
        """
        super(SoftmaxLearning, self).__init__(nb_actions)

        self.learning = learning
        self.temperature = temperature

    def actions(self, episode):
        values, error = self.learning.actions(episode)
        self.adjustTemperature(episode, error)

        # Exponentials
        vals = [_bounded_exp(v / self.temperature) for v in values]

        # Normalized to a softmax distribution
        return [v / sum(vals) for v in vals], error

    def adjustTemperature(self, episode, td_error):
        """ Dynamically adjust the Softmax temperature based on an episode (that
            has already been processed by the wrapped model) and its TD error.
        """
        pass
