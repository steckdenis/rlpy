from .historymodel import *

class KerasHistoryModel(HistoryModel):
    """ Base class for Keras-based recurrent neural networks
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
            batch_size=1,
            nb_epoch=4
        )
