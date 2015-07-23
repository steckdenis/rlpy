#
# Copyright (c) 2015 Vrije Universiteit Brussel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random

from .abstractworld import *
from .episode import *

class GridWorld(AbstractWorld):
    """ Grid of a given dimension with a starting position, a goal and an obstacle
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, width, height, initial, goal, obstacle, stochastic):
        """ Create a new grid world.

            @param width Width of the world, in number of cells
            @param height Height of the world
            @param initial (x, y) coordinates of the initial position
            @param goal (x, y) coordinates of the goal
            @param obstacle (x, y) coordinates of the obstacle
            @param stochastic True if noise has to be added to the actions taken
            @param polar True if the agent must only be able to sense its orientation
                   and distance to the closest wall
        """
        super(GridWorld, self).__init__()

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
        # The current position is set to the initial position
        self._current_pos = self.initial

        if self.stochastic:
            # The initial position is updated if the world is stochastic (so that
            # the next reset() call or episode will see the new initial position)
            self.initial = (random.randrange(self.width), random.randrange(self.height))

    def performAction(self, action):
        # Compute the coordinates of the candidate new position
        pos = self._current_pos

        if action == self.UP:
            pos = (pos[0], pos[1] - 1)
        elif action == self.DOWN:
            pos = (pos[0], pos[1] + 1)
        elif action == self.LEFT:
            pos = (pos[0] - 1, pos[1])
        elif action == self.RIGHT:
            pos = (pos[0] + 1, pos[1])

        # Check for the grid size, obstacle or goal
        if pos == self.goal:
            return (pos, 10.0, True)
        elif pos[0] < 0 or pos[1] < 0 or pos[0] >= self.width or pos[1] >= self.height or pos == self.obstacle:
            return (self._current_pos, -2.0, False)
        else:
            self._current_pos = pos

            return (pos, -1.0, False)
