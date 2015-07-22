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
    from keras.layers.recurrent import JZS1
except ImportError:
    print('Keras is not installed, do not use mut1model')

from .kerashistorymodel import *

class MUT1Model(KerasHistoryModel):
    """ Associate values to sequences of observations using the MUT1 model discovered
        by Jozefowicz et al,  2015.
    """

    def __init__(self, nb_actions, history_length, hidden_neurons):
        super(MUT1Model, self).__init__(nb_actions, history_length, hidden_neurons)

    def createKerasModel(self, state_size):
        """ Create an LSTM-based neural network
        """
        model = Sequential()
        model.add(JZS1(state_size, self.hidden_neurons, activation='tanh', inner_activation='tanh', truncate_gradient=self.history_length))
        model.add(Dense(self.hidden_neurons, self.nb_actions, activation='linear'))

        return model
