from numpy import zeros, array, transpose, float32

try:
    import clstm
except ImportError:
    print('clstm is not installed, CLSTMModel is therefore unavailable')

from .abstractmodel import *

class CLSTMModel(AbstractModel):
    """ Associate values to sequences using an LSTM network based on the clstm library
    """

    def __init__(self, nb_actions, hidden_neurons):
        """ Constructor.

            @param hidden_neurons Number of neurons in the hidden layer
        """
        super(CLSTMModel, self).__init__(nb_actions)

        self.hidden_neurons = hidden_neurons
        self._values = zeros(shape=(1, 1, 1), dtype=float32)
        self._model = None

    def values(self, episode):
        # Make the prediction
        if self._model is None:
            value = [0.0] * self.nb_actions
        else:
            timesteps = len(episode.states)
            state_size = len(episode.states[0])

            # Put the states of the episode in an array of the correct shape
            data = zeros(shape=(timesteps, state_size, 1), dtype=float32)

            data[:, :, 0] = episode.states

            # Pass the data to the model
            self._model.inputs.aset(data)
            self._model.forward()

            # Return what the model predicted for the batch, all the variables
            # and the last time step
            clstm.array_of_sequence(self._values, self._model.outputs)

            value = list(self._values[-1, :, 0])

        return value

    def learn(self, episodes):
        state_size = len(episodes[0].states[0])

        # Create the model of this action if it does not exist yet
        if self._model is None:
            self._model = clstm.make_net_init(
                "lstm1",
                "ninput=%i:nhidden=%i:noutput=%i:output_type=LinearLayer" % \
                    (state_size, self.hidden_neurons, self.nb_actions)
            )

        # Create an (timestep, variable, batch) array containing the states
        max_timestep = max([len(episode.states) for episode in episodes])
        data = zeros(shape=(max_timestep, state_size, len(episodes)), dtype=float32)
        values = zeros(shape=(max_timestep, self.nb_actions, len(episodes)), dtype=float32)

        for e, episode in enumerate(episodes):
            # Number of timesteps of this episode
            timesteps = len(episode.states)

            # episode.states has a shape of (timestep, variable), which corresponds
            # to what we want to put in the data array.
            data[0:timesteps, :, e] = episode.states

            # episode.states has a shape of (timestep, action), which also corresponds
            # to the desired shape
            values[0:timesteps, :, e] = episode.values

        # Train the model
        print('training')
        self._model.inputs.aset(data)
        self._model.forward()

        errors = values - self._model.outputs.array()
        self._model.d_outputs.aset(errors)
        self._model.backward()
        self._model.update()
        print('done')
