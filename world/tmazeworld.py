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

class TMazeWorld(AbstractWorld):
    """ Long corridor followed by a T junction. The agent senses during the first
        few steps an information about which direction to take at the T junction
        in order to get the reward, and this information disappears afterwards. This
        makes this world partially observable.
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, length, info_time):
        """ Create a new grid world.

            @param length Length of the corridor
            @param info_time Time during which the information about which direction
                             to take is available.
        """
        super(TMazeWorld, self).__init__()

        self.length = length
        self.info_time = info_time

        self.reset()

    def nb_actions(self):
        return 4

    def reset(self):
        # The current position is set to the initial position
        self._current_pos = 0
        self._timestep = 0
        self._target_dir = random.choice([self.UP, self.DOWN])

        self.initial = self.makeState()

    def performAction(self, action):
        # Increment the current timestep, so that the information can be given
        # only for a limited amount of time
        self._timestep += 1

        # Compute the coordinates of the candidate new position, using the same
        # algorithm as in the gridworld
        pos = (self._current_pos, 0)

        if action == self.UP:
            pos = (pos[0], -1)
        elif action == self.DOWN:
            pos = (pos[0], 1)
        elif action == self.LEFT:
            pos = (pos[0] - 1, 0)
        elif action == self.RIGHT:
            pos = (pos[0] + 1, 0)

        # Check the validity of the position
        if pos[0] == self.length - 1 and pos[1] == -1:
            # Up part of the T junction
            self._current_pos = pos[0]

            return (self.makeState(), 10.0 if self._target_dir == self.UP else 0.0, True)
        elif pos[0] == self.length - 1 and pos[1] == 1:
            # Down part of the T junction
            self._current_pos = pos[0]

            return (self.makeState(), 10.0 if self._target_dir == self.DOWN else 0.0, True)
        elif pos[1] == -1 or pos[1] == 1 or pos[0] < 0 or pos[0] >= self.length:
            # Overflow
            return (self.makeState(), -2.0, False)
        else:
            # Simple movement
            self._current_pos = pos[0]

            return (self.makeState(), -1.0, False)

    def makeState(self):
        if self._timestep <= self.info_time:
            # Provide information about the target direction
            return (self._current_pos, self._target_dir + 1)
        else:
            # Provide no hint
            return (self._current_pos, 0)
