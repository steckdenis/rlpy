from abstractmodel import *

class OneOfNModel(AbstractModel):
    """ Take a discrete state (where each variable is an integer of finite range)
        and transforms it to a vector of ones and zeroes.

        [4, 5] -> [0, 0, 0, 0, 1, 0 ... , 0, 0, 0, 0, 0, 1, 0, ...]
    """

    def __init__(self, nb_actions, model, ranges):
        """ Constructor.

            @param model Model that is wrapped by this model
            @param ranges List of maximum values that the input state can take, 
                          for instance (10, 5) for a 10 by 5 grid.
        """
        super().__init__(nb_actions)

        self.model = model
        self.ranges = ranges
        self.output_dim = sum(ranges)

    def nextEpisode(self):
        self.model.nextEpisode()

    def values(self, state):
        return self.model.values(self.make_state(state))

    def setValue(self, state, action, value):
        self.model.setValue(self.make_state(state), action, value)

    def make_state(self, state):
        """ Transform a state as explained in the docstring of this class
        """
        res = [0.0] * self.output_dim
        offset = 0

        for index, value in enumerate(state):
            res[offset + int(value)] = 1.0
            offset += self.ranges[index]

        return tuple(res)