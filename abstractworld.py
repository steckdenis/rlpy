class AbstractWorld(object):
    """ Abstract world that receives actions and produces new states
    """

    def nb_actions(self):
        """ Return the number of actions that can be performed. This number
            cannot change during the lifetime of the world.
        """
        raise NotImplementedError('The world does not implement nb_actions()')

    def reset(self):
        """ Reset the world in its original configuration, as if no agent performed
            actions on it.
        """
        raise NotImplementedError('The world does not implement reset()')

    def performAction(self, action):
        """ Perform an action on the world and return a tuple (state, reward, finished)
        """
        raise NotImplementedError('The world does not implement performAction()')