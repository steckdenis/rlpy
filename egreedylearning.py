import random

from abstractlearning import *

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

    def action(self, state, last_reward):
        # Compute the exploitation step, and allow the learning algorithm to
        # keep track of the reward
        action = self.learning.action(state, last_reward)

        if random.random() < self.epsilon:
            # Exploration step
            return random.randrange(0, self.nb_actions)
        else:
            # Exploitation step
            return action