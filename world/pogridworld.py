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

class POGridWorld(GridWorld):
    """ Grid of a given dimension with a starting position, a goal and an obstacle.
        The agent can only sense its x coordinate
    """

    def __init__(self, width, height, initial, goal, obstacle, stochastic):
        """ Create a new grid world.
        """
        super(POGridWorld, self).__init__(width, height, initial, goal, obstacle, stochastic)

    def performAction(self, action):
        # Normal action in the gridworld
        (pos, reward, finished) = super(POGridWorld, self).performAction(action)

        # Set Y to zero, so that only X can be observed but the state dimension
        # remains compatible with GridWorld
        return ((pos[0], 0.0), reward, finished)