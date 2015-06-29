from abstractmodel import *

class DiscreteModel(AbstractModel):
    """ Model used for storing values associated with discrete states, with no
        function approximation.
    """

    def __init__(self, nb_actions):
        super().__init__(nb_actions)

        self._data = {}

    def values(self, state):
        return [self._data.get(self.key(state, action), 0.0) for action in range(self.nb_actions)]

    def setValue(self, state, action, value):
        self._data[self.key(state, action)] = value

    def key(self, state, action):
        return tuple(state + (action,))