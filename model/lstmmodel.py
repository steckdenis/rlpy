from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from numpy import zeros, array, float32

from .historymodel import *

class LSTMModel(HistoryModel):
    """ Associate values to sequences of observations using a neural network
        made of standard perceptron-like layers and LSTM memory cells.
    """

    def __init__(self, nb_actions, history_length, hidden_neurons):
        """ Constructor.

            @param hidden_neurons Number of neurons in the hidden layer
        """
        super().__init__(nb_actions, history_length)

        self.hidden_neurons = hidden_neurons

    def createModel(self, state_size):
        """ Create an LSTM-based neural network
        """
        model = Sequential()
        model.add(LSTM(state_size, self.nb_actions, activation='linear', inner_activation='sigmoid'))

        print('Compiling model...')
        model.compile(loss='mse', optimizer='rmsprop')
        print('Compiled')

        return model

    def getValues(self, observations):
        """ Predict the value of one sequence of observations
        """
        return self._model.predict(observations, verbose=0)[0]

    def trainModel(self, data, values):
        """ Train the model with data and values.

            @param data (sequences, max_length, state_size) array representing
                        the observed states.
            @param values (sequences, nb_actions) array representing the values
                          associated with the observed sequences.
        """
        self._model.fit(
            data,
            values,
            verbose=0,
            batch_size=10,
            nb_epoch=100
        )
