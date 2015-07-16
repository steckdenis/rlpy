from .abstractmodel import *

class DiscreteModel(AbstractModel):
    """ Model used for storing values associated with discrete states, with no
        function approximation.
    """

    def __init__(self, nb_actions):
        super(DiscreteModel, self).__init__(nb_actions)

        self._data = {}

    def values(self, episode):
        """ Return the values of the last state of an episode
        """
        state = episode.states[-1]

        return [self._data.get(self.key(state, action), 0.0) for action in range(self.nb_actions)]

    def learn(self, episodes):
        """ Update the model using the updated values in the episodes.
        """
        for episode in episodes:
            for state, action, values in zip(episode.states, episode.actions, episode.values):
                self._data[self.key(state, action)] = values[action]

    def key(self, state, action):
        return tuple(state + (action,))