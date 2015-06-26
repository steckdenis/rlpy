import random

from abstractworld import *

class GridWorld(AbstractWorld):
    """ Grid of a given dimension with a starting position, a goal and an obstacle
    """

    def __init__(self, width, height, initial, goal, obstacle, stochastic):
        """ Create a new grid world.

            @param width Width of the world, in number of cells
            @param height Height of the world
            @param initial (x, y) coordinates of the initial position
            @param goal (x, y) coordinates of the goal
            @param obstacle (x, y) coordinates of the obstacle
        """

        self.width = width
        self.height = height
        self.initial = initial
        self.goal = goal
        self.obstacle = obstacle
        self.stochastic = stochastic

        self.reset()

    def nb_actions(self):
        return 4

    def reset(self):
        self._current_pos = self.initial

    def performAction(self, action):
        # Compute the coordinates of the candidate new position
        pos = self._current_pos

        # If stochasticity is enabled, perturb the action that will be performed
        if self.stochastic and random.random() < 0.2:
            action = random.randint(0, 3)

        if action == 0:
            # UP
            pos = (pos[0], pos[1] - 1)
        elif action == 1:
            # DOWN
            pos = (pos[0], pos[1] + 1)
        elif action == 2:
            # LEFT
            pos = (pos[0] - 1, pos[1])
        elif action == 3:
            # RIGHT
            pos = (pos[0] + 1, pos[1])

        # Check for the grid size, obstacle or goal
        if pos == self.goal:
            return (pos, 10, True)
        elif pos[0] < 0 or pos[1] < 0 or pos[0] >= self.width or pos[1] >= self.height or pos == self.obstacle:
            return (self._current_pos, -2, False)
        else:
            self._current_pos = pos

            return (pos, -1, False)