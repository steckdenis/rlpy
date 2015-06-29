class AbstractModel(object):
    """ Model used to associate values with keys
    """

    def __init__(self, nb_actions):
        """ Constructor.

            @param nb_actions Number of possible actions in the world
        """
        self.nb_actions = nb_actions

    def setValue(self, state, action, value):
        """ Set a value associated with a state and an action. If no concept
            of action is relevant, setting this parameter to -1 allows the model
            to disable its handling of actions
        """
        raise NotImplementedError('The model does not implement setValue()')

    def values(self, state):
        """ Return the values associated with a state, for each action.
        """
        raise NotImplementedError('The model does not implement value()')

    def nextEpisode(self):
        """ Called every time the agent is reset and a new episode is started.
            Data that has been learned by the model should not be discarded, but
            history-based models can use this information in order to flush their
            history.
        """
        pass