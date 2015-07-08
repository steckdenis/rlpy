from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from numpy import zeros, array, float32

from .abstractmodel import *

class LSTMModel(AbstractModel):
    """ Model used for storing values associated with discrete states, with no
        function approximation.
    """

    def __init__(self, nb_actions, history_length):
        """ Constructor.

            @param history_length The last @p history_length observations of every
                                  subsequence of observations are used to predict
                                  a value. For instance, the network may learn
                                  an application o1o2o3 -> v1, o2o3o4 -> v2, etc.
        """
        super().__init__(nb_actions)

        self.history_length = history_length
        self._model = None

    def values(self, episode):
        # Make the prediction
        if self._model is None:
            value = [0.0] * self.nb_actions
        else:
            nb_states = min(len(episode.states), self.history_length)

            value = self._model.predict(self.make_data([episode.states[-nb_states:]]), verbose=0)[0]

        return value

    def learn(self, episodes):
        state_size = len(episodes[0].states[0])

        # Create the model of this action if it does not exist yet
        if self._model is None:
            self._model = Sequential()
            self._model.add(LSTM(state_size, self.nb_actions, activation='linear', inner_activation='sigmoid'))

            print('Compiling model...')
            self._model.compile(loss='mse', optimizer='rmsprop')
            print('Compiled')

        # Create an (total states encountered, history_length, state_dim) array
        total_length = sum([len(episode.states) for episode in episodes])

        data = zeros(shape=(total_length, self.history_length, state_size), dtype=float32)
        values = []
        i = 0

        for e, episode in enumerate(episodes):
            for t in range(len(episode.states)):
                # Observations t-history_length..t of the episode, and the value
                # that this sequence has to produce
                length = min(t + 1, self.history_length)

                data[i, 0:length, :] = episode.states[t + 1 - length:t + 1]
                values.append(episode.values[t])

        # Train the model : the sequence of states 0 to n-1 must return the value v[-2],
        # the one that has just been updated
        print('Training LSTM')
        for i in range(5):
            self._model.train(
                data,
                self.make_data(values)
            )
        print('done')

    def make_data(self, data):
        """ Return an ndarray having row per element in data and one column
            per element of data[:]
        """

        return array(data, dtype=float32)