import collections

MAX_EPISODE_LENGTH = 100

class Episode(object):
    """ Sequence of actions and observations that correspond to a learning episode
    """

    def __init__(self):
        self.states = collections.deque([], MAX_EPISODE_LENGTH)
        self.values = collections.deque([], MAX_EPISODE_LENGTH)
        self.actions = collections.deque([], MAX_EPISODE_LENGTH)
        self.rewards = collections.deque([], MAX_EPISODE_LENGTH)

    def addState(self, state):
        """ Add a state observation to the episode
        """
        self.states.append(state)

    def addAction(self, action):
        """ Add an action to the episode. This allows the agent to keep track of
            its past actions.
        """
        self.actions.append(action)

    def addReward(self, reward):
        """ Add a reward to the episode. This is an instantaneous reward received
            after having performed an action.
        """
        self.rewards.append(reward)

    def addValues(self, values):
        """ Add values for actions of the last state.
        """
        self.values.append(values)