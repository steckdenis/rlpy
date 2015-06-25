from abstractmodel import *

class DiscreteModel(AbstractModel):
    """ Model used for storing values associated with discrete states, with no
        function approximation.
    """

    def __init__(self):
        self._data = {}

    def value(self, state, action):
        return self._data.get(self.key(state, action), 0.0)

    def setValue(self, state, action, value):
        self._data[self.key(state, action)] = value

    def key(self, state, action):
        return tuple(state + (action,))