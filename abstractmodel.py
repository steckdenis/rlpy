class AbstractModel(object):
    """ Model used to associate values with keys
    """

    def setValue(state, action, value):
        """ Set a value associated with a state and an action. If no concept
            of action is relevant, setting this parameter to -1 allows the model
            to disable its handling of actions
        """
        raise NotImplementedError('The model does not implement setValue()')

    def value(state, action):
        """ Return the value associated with a state-action. Action can be set
            to -1 if no concept of action is relevant for the caller.
        """
        raise NotImplementedError('The model does not implement value()')