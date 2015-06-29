from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from numpy import ndarray, array, float32

from abstractmodel import *

class LSTMModel(AbstractModel):
    """ Model used for storing values associated with discrete states, with no
        function approximation.
    """

    def __init__(self, nb_actions):
        super().__init__(nb_actions)

        self._model = None

        self.nextEpisode()

    def nextEpisode(self):
        # Clear the lists of states used to query or feed the network
        self._states = []               # List of states queried for prediction
        self._values = []               # Values predicted (or updated for training)

    def values(self, state):
        # Add the required state to the list of states visited for prediction
        self._states.append(state)

        # Make the prediction
        if self._model is None:
            value = [0.0] * self.nb_actions
        else:
            value = self._model.predict(self.make_data([self._states]), verbose=0)[0]

        self._values.append(value)
        return value

    def setValue(self, state, action, value):
        # This model only allows the value associated with the last but one state to be changed
        assert(state == self._states[-2])

        # Create the model of this action if it does not exist yet
        if self._model is None:
            self._model = Sequential()
            self._model.add(LSTM(len(state), self.nb_actions, activation='linear', inner_activation='hard_sigmoid'))

            print('Compiling model...')
            self._model.compile(loss='mse', optimizer='sgd') # rmsprop
            print('Compiled')

        # Update the data
        self._values[-2][action] = value

        # Train the model : the sequence of states 0 to n-1 must return the value v[-2],
        # the one that has just been updated
        self._model.train(self.make_data([self._states[:-1]]), self.make_data(self._values[-2:-1]))

    def make_data(self, data):
        """ Return an ndarray having row per element in data and one column
            per element of data[:]
        """

        return array(data, dtype=float32)