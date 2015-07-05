from abstractlearning import *

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
        super().__init__(nb_actions)

        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa

    def action(self, episode):
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

        # Choose the best action, the one with the highest advantage value
        action = None
        action_A = 0

        for a, A in enumerate(episode.values[-1]):
            if action is None or A > action_A:
                action = a
                action_A = A

        return action