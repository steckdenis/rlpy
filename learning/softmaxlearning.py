import math

from .abstractlearning import *

def _bounded_exp(x):
    """ Exponential with clamping of big and small X values
    """
    if x < -20.0:
        return 1e-9
    elif x > 30.0:
        return 1e13
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
        super().__init__(nb_actions)

        self.learning = learning
        self.temperature = temperature

    def actions(self, episode):
        values = self.learning.actions(episode)

        # Exponentials
        vals = [_bounded_exp(v / self.temperature) for v in values]

        # Normalized to a softmax distribution
        return [v / sum(vals) for v in vals]