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

from .gridworld import *

class PolarGridWorld(GridWorld):
    """ Grid of a given dimension with a starting position, a goal and an obstacle.
        The agent is only able to sense its direction and distance to the wall
        in front of it.
    """
    TURN_LEFT = 0
    TURN_RIGHT = 1
    GO_FORWARD = 2
    GO_BACKWARD = 3

    def __init__(self, width, height, initial, goal, obstacle, stochastic):
        """ Create a new grid world.
        """
        super(PolarGridWorld, self).__init__(width, height, initial, goal, obstacle, stochastic)

    def reset(self):
        super(PolarGridWorld, self).reset()

        self._current_dir = self.RIGHT

    def performAction(self, action):
        # Compute the coordinates of the candidate new position
        pos = self._current_pos

        # Turning does not change the position
        if action == self.TURN_LEFT:
            self._current_dir = (self._current_dir + 1) % 4
        elif action == self.TURN_RIGHT:
            self._current_dir = (self._current_dir - 1) % 4
        else:
            # Compute the new position
            offset = 1 if action == self.GO_FORWARD else -1

            if self._current_dir == self.UP:
                pos = (pos[0], pos[1] - offset)
            elif self._current_dir == self.DOWN:
                pos = (pos[0], pos[1] + offset)
            elif self._current_dir == self.LEFT:
                pos = (pos[0] - offset, pos[1])
            elif self._current_dir == self.RIGHT:
                pos = (pos[0] + offset, pos[1])

        # Check for the grid size, obstacle or goal
        finished = False

        if pos[0] < 0 or pos[1] < 0 or pos[0] >= self.width or pos[1] >= self.height or pos == self.obstacle:
            reward = -2.0
        else:
            self._current_pos = pos
            finished = (pos == self.goal)
            reward = 10.0 if finished else -1.0

        # Compute the distance from the wall in front of the agent
        if self._current_dir == self.UP:
            distance = pos[1]
        elif self._current_dir == self.DOWN:
            distance = self.height - pos[1] - 1
        elif self._current_dir == self.LEFT:
            distance = pos[0]
        elif self._current_dir == self.RIGHT:
            distance = self.width - pos[0] - 1

        return ((distance, self._current_dir), reward, finished)