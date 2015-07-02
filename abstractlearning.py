class AbstractLearning(object):
    """ Abstract class for learning strategies. Instances of this class receive
        state observations and rewards and have to choose the action to perform.
    """

    def __init__(self, nb_actions):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
        """

        self.nb_actions = nb_actions

    def action(self, episode):
        """ Return the action index that should be performed given an history
            (represented by an Episode object)

            @return An integer from 0 to nb_actions-1

            @note No explaratory steps should be returned by the learning strategy,
                  except if the strategy consist of adding exploratory steps to
                  another one (EgreedyLearning(QLearning()) for instance).
        """
        raise NotImplementedError('The learning strategy does not implement action()')