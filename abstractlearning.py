class AbstractLearning(object):
    """ Abstract class for learning strategies. Instances of this class receive
        state observations and rewards and have to choose the action to perform.
    """

    def __init__(self, nb_actions):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
        """

        self.nb_actions = nb_actions

    def actions(self, episode):
        """ Return a probability density over the actions that should be performed
            given an history (represented by an Episode object)

            @return A list of nb_actions elements, each value giving the probability
                    that the corresponding action has to be taken.
        """
        raise NotImplementedError('The learning strategy does not implement action()')