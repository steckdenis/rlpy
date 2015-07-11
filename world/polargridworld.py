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
        super().__init__(width, height, initial, goal, obstacle, stochastic)

    def reset(self):
        self._current_pos = self.initial
        self._current_dir = self.RIGHT

    def performAction(self, action):
        # Compute the coordinates of the candidate new position
        pos = self._current_pos

        # If stochasticity is enabled, perturb the action that will be performed
        if self.stochastic and random.random() < 0.2:
            action = random.randint(0, 3)

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