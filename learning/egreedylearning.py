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
        super().__init__(nb_actions)

        self.learning = learning
        self.epsilon = epsilon

    def actions(self, episode):
        # The best action has a probability 1-epsilon to be taken, the others share
        # a probability of epsilon
        actions = self.learning.actions(episode)

        best_index = 0
        best_proba = -1000000.0

        for index, a in enumerate(actions):
            if a > best_proba:
                best_proba = a
                best_index = index

        actions = [self.epsilon / (self.nb_actions - 1) for a in actions]
        actions[best_index] = 1.0 - self.epsilon

        return actions