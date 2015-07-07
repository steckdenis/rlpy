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
        super().__init__(nb_actions)

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
            batch_size=10,
            nb_epoch=100
        )

    def make_data(self, data):
        """ Return an ndarray having row per element in data and one column
            per element of data[:]
        """
        return array(data, dtype=float32)