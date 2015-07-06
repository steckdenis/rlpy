from .abstractworld import *

class OneOfNWorld(AbstractWorld):
    """ Take a discrete state (where each variable is an integer of finite range)
        and transforms it to a vector of ones and zeroes.

        [4, 5] -> [0, 0, 0, 0, 1, 0 ... , 0, 0, 0, 0, 0, 1, 0, ...]
    """

    def __init__(self, world, ranges):
        """ Constructor.

            @param ranges List of maximum values that the input state can take, 
                          for instance (10, 5) for a 10 by 5 grid.
        """

        self.world = world
        self.ranges = ranges
        self.output_dim = sum(ranges)

        self.initial = self.make_state(world.initial)

    def nb_actions(self):
        return self.world.nb_actions()

    def reset(self):
        self.world.reset()

    def performAction(self, action):
        state, reward, finished = self.world.performAction(action)

        # Adjust the state returned by the world
        state = self.make_state(state)

        return (state, reward, finished)

    def plotModel(self, model):
        self.world.plotModel(model)

    def make_state(self, state):
        """ Transform a state as explained in the docstring of this class
        """
        res = [0.0] * self.output_dim
        offset = 0

        for index, value in enumerate(state):
            res[offset + int(value)] = 1.0
            offset += self.ranges[index]

        return tuple(res)