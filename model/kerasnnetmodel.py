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

from numpy import ndarray, array, float32

try:
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout
except ImportError:
    print('Keras is not installed, do not use kerasnnetmodel')

from .abstractmodel import *

class KerasNnetModel(AbstractModel):
    """ Simple perceptron with a single hidden layer (using Keras)
    """

    def __init__(self, nb_actions, hidden_neurons):
        super(KerasNnetModel, self).__init__(nb_actions)

        self.hidden_neurons = hidden_neurons
        self._model = None

    def values(self, episode):
        # Make the prediction if the model is available
        if self._model is None:
            value = [0.0] * self.nb_actions
        else:
            state = episode.states[-1]
            value = self._model.predict(self.make_data([state]), verbose=0)[0]

        return value

    def learn(self, episodes):
        # Create the model if needed
        if self._model is None:
            self._model = Sequential()

            self._model.add(Dense(len(episodes[0].states[0]), self.hidden_neurons, init='uniform', activation='tanh'))
            self._model.add(Dense(self.hidden_neurons, self.nb_actions, init='uniform', activation='linear'))

            print('Compiling model...')
            self._model.compile(loss='mse', optimizer='rmsprop')
            print('Compiled')

        # Store the values of all the states encountered in all the episodes
        states = []
        values = []

        for episode in episodes:
            states.extend(episode.states)
            values.extend(episode.values)

        # Train for these values
        self._model.fit(
            self.make_data(states),
            self.make_data(values),
            verbose=0,
            batch_size=20,
            nb_epoch=2
        )

    def make_data(self, data):
        """ Return an ndarray having row per element in data and one column
            per element of data[:]
        """
        return array(data, dtype=float32)
