from numpy import zeros, array, float32

try:
    from keras.models import Sequential
    from keras.layers.core import Dense, TimeDistributedDense
    from keras.layers.recurrent import LSTM
except ImportError:
    print('Keras is not installed, do not use lstmmodel')

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
        model.add(LSTM(state_size, self.hidden_neurons, activation='tanh', inner_activation='linear', truncate_gradient=self.history_length))
        model.add(Dense(self.hidden_neurons, self.nb_actions, activation='linear'))

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
            verbose=1,
            batch_size=1,
            nb_epoch=2
        )
