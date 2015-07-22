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
    from fann2 import libfann
except ImportError:
    try:
        from pyfann import libfann
    except ImportError:
        print('FANN is not installed, do not use fannnnetmodel')

from .abstractmodel import *

class FannNnetModel(AbstractModel):
    """ Simple perceptron with a single hidden layer (using py-FANN)
    """

    def __init__(self, nb_actions, hidden_neurons):
        super(FannNnetModel, self).__init__(nb_actions)

        self.hidden_neurons = hidden_neurons
        self._model = None

    def values(self, episode):
        # Make the prediction if the model is available
        if self._model is None:
            value = [0.0] * self.nb_actions
        else:
            state = episode.states[-1]
            value = self._model.run(state)

        return value

    def learn(self, episodes):
        state_size = len(episodes[0].states[0])

        # Create the model if needed
        if self._model is None:
            self._model = libfann.neural_net()
            self._model.create_sparse_array(1, (state_size, self.hidden_neurons, self.nb_actions))
            self._model.randomize_weights(-0.1, 0.1)
            self._model.set_activation_function_layer(libfann.GAUSSIAN, 1)
            self._model.set_activation_function_layer(libfann.LINEAR, 2)

        # Store the values of all the states encountered in all the episodes
        states = []
        values = []

        for episode in episodes:
            states.extend(episode.states)
            values.extend(episode.values)

        # Train for these values
        data = libfann.training_data()
        data.set_train_data(states, values)

        self._model.train_on_data(data, 150, 50, 1e-5)

    def make_data(self, data):
        """ Return an ndarray having row per element in data and one column
            per element of data[:]
        """
        return array(data, dtype=float32)
