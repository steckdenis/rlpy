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
from numpy import array, float32

from .softmaxlearning import *

try:
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout
except ImportError:
    print('Keras is not installed, AdaptiveSoftmaxLearning is unavailable')

class AdaptiveSoftmaxLearning(SoftmaxLearning):
    """ Softmax action selection that increases its temperature for states having
        a big TD error.
    """
    def __init__(self, nb_actions, learning, hidden_neurons, discount_factor):
        super(AdaptiveSoftmaxLearning, self).__init__(nb_actions, learning, 1.0)

        self.hidden_neurons = hidden_neurons
        self.discount_factor = discount_factor

        self._model = None
        self._states = []
        self._values = []

    def adjustTemperature(self, episode, td_error):
        # Create the model for the TD error when we know the state-space size
        if self._model is None:
            self._model = Sequential()

            self._model.add(Dense(len(episode.states[0]), self.hidden_neurons, init='uniform', activation='tanh'))
            self._model.add(Dense(self.hidden_neurons, 1, init='uniform', activation='linear'))

            print('Compiling TD-error model...')
            self._model.compile(loss='mse', optimizer='rmsprop')
            print('Compiled')

        # Compute the new temperature : y(t) = |td_error| + beta*y(t+1)
        #
        # Formula given in "Reinforcement Learning with Long Short-Term Memory",
        # Bram Bakker, 2001
        #
        # Because t+1 is not yet known, another formula is used: the model
        # is used to predict y(t), and y(t-1) = |td_error| + beta*y(t) is used
        # to train the model for the previous observation
        current_state = episode.states[-1]
        current_temperature = self._model.predict(self.make_data([current_state]))[0][0]

        updated_temperature = abs(td_error) + self.discount_factor * current_temperature

        if len(episode.states) > 1:
            # Update the model
            previous_state = episode.states[-2]

            self._states.append(previous_state)
            self._values.append([updated_temperature])

            # Learn in batch for speed and accuracy
            if len(self._states) > 100:
                self._model.fit(
                    self.make_data(self._states),
                    self.make_data(self._values),
                    verbose=0,
                    batch_size=10,
                    nb_epoch=10
                )

                # Stats
                temps = [v[0] for v in self._values]
                print('Average temperature for batch', sum(temps) / len(temps))

                self._states = []
                self._values = []

        # Use the new tempoerature (without allowing it to be too small)
        self.temperature = max(current_temperature, 0.2)

    def make_data(self, data):
        """ Return an ndarray having row per element in data and one column
            per element of data[:]
        """
        return array(data, dtype=float32)
