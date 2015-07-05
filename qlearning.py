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

    def action(self, episode):
        """ Return the action index that should be performed given a given state
            observation and the reward received for last action.

            @return An integer from 0 to nb_actions-1
        """

        # Update the Q-value of the last action that was taken
        if len(episode.actions) > 0:
            last_action = episode.actions[-1]
            last_reward = episode.rewards[-1]

            Q = episode.values[-2][last_action]
            error = last_reward + self.gamma * max(episode.values[-1]) - Q

            episode.values[-2][last_action] = Q + self.alpha * error

        # Choose the best action
        action = None
        action_Q = 0

        for a, Q in enumerate(episode.values[-1]):
            if action is None or Q > action_Q:
                action = a
                action_Q = Q

        return action