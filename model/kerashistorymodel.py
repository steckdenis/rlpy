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

from .historymodel import *

class KerasHistoryModel(HistoryModel):
    """ Base class for Keras-based recurrent neural networks
    """

    def __init__(self, nb_actions, history_length, hidden_neurons):
        """ Constructor.

            @param hidden_neurons Number of neurons in the hidden layer
        """
        super(KerasHistoryModel, self).__init__(nb_actions, history_length)

        self.hidden_neurons = hidden_neurons

    def createModel(self, state_size):
        """ Create an LSTM-based neural network
        """
        model = self.createKerasModel(state_size)

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
            nb_epoch=4
        )
