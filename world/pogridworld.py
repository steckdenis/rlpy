import random

from .gridworld import *

class POGridWorld(GridWorld):
    """ Grid of a given dimension with a starting position, a goal and an obstacle.
        The agent can only sense its x coordinate
    """

    def __init__(self, width, height, initial, goal, obstacle, stochastic):
        """ Create a new grid world.
        """
        super().__init__(width, height, initial, goal, obstacle, stochastic)

    def performAction(self, action):
        # Normal action in the gridworld
        (pos, reward, finished) = super().performAction(action)

        return ((pos[0],), reward, finished)