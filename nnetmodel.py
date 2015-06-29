from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from numpy import ndarray, array, float32

from abstractmodel import *

class NnetModel(AbstractModel):
    """ Simple perceptron with a single hidden layer
    """

    def __init__(self, nb_actions, hidden_neurons):
        super().__init__(nb_actions)

        self.hidden_neurons = hidden_neurons
        self._model = None

        self.nextEpisode()

    def nextEpisode(self):
        self._states = []       # States visited by values()
        self._values = []       # Values returned by values()

    def values(self, state):
        # Make the prediction if the model is available
        if self._model is None:
            value = [0.0] * self.nb_actions
        else:
            value = self._model.predict(self.make_data([state]), verbose=0)[0]

        # Keep track of the value predicted, so that it can be updated by setValue
        self._states.append(state)
        self._values.append(value)

        return value

    def setValue(self, state, action, value):
        # This model only allows the value associated with the last but one state to be changed
        assert(state == self._states[-2])

        # Update the last-but-one value (the last value has been returned by
        # values() and is not the one that the learning algorithm is updating)
        self._values[-2][action] = value

        # Retrain the model
        if self._model is None:
            # New perceptron model
            self._model = Sequential()

            self._model.add(Dense(len(self._states[0]), self.hidden_neurons, init='uniform', activation='linear'))
            self._model.add(Dense(self.hidden_neurons, self.nb_actions, init='uniform', activation='linear'))

            print('Compiling model...')
            self._model.compile(loss='mse', optimizer='sgd') # rmsprop
            print('Compiled')

        for i in range(5):
            # Perform 5 gradient updates (not more, because we would unlearn the previous episodes)
            self._model.train(self.make_data(self._states[-2:-1]), self.make_data(self._values[-2:-1]))

    def make_data(self, data):
        """ Return an ndarray having row per element in data and one column
            per element of data[:]
        """
        return array(data, dtype=float32)