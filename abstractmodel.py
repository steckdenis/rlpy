class AbstractModel(object):
    """ Model used to associate values with keys
    """

    def __init__(self, nb_actions):
        """ Constructor.

            @param nb_actions Number of possible actions in the world
        """
        self.nb_actions = nb_actions

    def learn(self, episodes):
        """ Update the model using the episodes provided. Each episode contains
            information about the states, actions and rewards visited by the agent.
        """
        raise NotImplementedError('The model does not implement learn()')

    def values(self, episode):
        """ Return the values associated with the last state of an episode
        """
        raise NotImplementedError('The model does not implement values()')
