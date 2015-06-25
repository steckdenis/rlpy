from abstractlearning import *

class QLearning(AbstractLearning):
    """ Q-Learning learning strategy
    """

    def __init__(self, nb_actions, model, alpha, gamma):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
            @param model Model used to store the Q-values. Can be any discrete or
                         continuous model. 
            @param alpha Learning factor
            @param gamma Discount factor
        """
        super().__init__(nb_actions)

        self.model = model
        self.alpha = alpha
        self.gamma = gamma

        self._last_action = None
        self._last_state = None

    def action(self, state, last_reward):
        """ Return the action index that should be performed given a given state
            observation and the reward received for last action.

            @return An integer from 0 to nb_actions-1
        """

        # Update the Q-value of the last action that was taken
        if self._last_action is not None:
            Q = self.model.value(self._last_state, self._last_action)

            Q += self.alpha * (
                last_reward +
                self.gamma * max([self.model.value(state, a) for a in range(self.nb_actions)]) -
                Q
            )

            self.model.setValue(self._last_state, self._last_action, Q)

        # Choose the best action
        action = None
        action_Q = 0

        for a in range(self.nb_actions):
            Q = self.model.value(state, a)

            if action is None or Q > action_Q:
                action = a
                action_Q = Q

        # Update some state information
        self._last_action = action
        self._last_state = state

        return action