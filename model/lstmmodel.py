#
# Copyright (c) 2015 Vrije Universiteit Brussel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
        super(LSTMModel, self).__init__(nb_actions, history_length, hidden_neurons)

    def createKerasModel(self, state_size):
        """ Create an LSTM-based neural network
        """
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, input_dim=state_size, activation='tanh', inner_activation='linear'))
        model.add(Dense(self.nb_actions, activation='linear'))

        return model
