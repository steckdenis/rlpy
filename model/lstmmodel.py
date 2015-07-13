try:
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.layers.recurrent import LSTM
except ImportError:
    print('Keras is not installed, do not use lstmmodel')

from .kerashistorymodel import *

class LSTMModel(KerasHistoryModel):
    """ Associate values to sequences of observations using a neural network
        made of standard perceptron-like layers and LSTM memory cells.
    """

    def __init__(self, nb_actions, history_length, hidden_neurons):
        super().__init__(nb_actions, history_length, hidden_neurons)

    def createKerasModel(self, state_size):
        """ Create an LSTM-based neural network
        """
        model = Sequential()
        model.add(LSTM(state_size, self.hidden_neurons, activation='tanh', inner_activation='linear', truncate_gradient=self.history_length))
        model.add(Dense(self.hidden_neurons, self.nb_actions, activation='linear'))

        return model
