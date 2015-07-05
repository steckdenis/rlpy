from abstractlearning import *

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
        super().__init__(nb_actions)

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