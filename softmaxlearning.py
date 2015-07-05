import math

from abstractlearning import *

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
        vals = [math.exp(v / self.temperature) for v in values]

        # Normalized to a softmax distribution
        return [v / sum(vals) for v in vals]