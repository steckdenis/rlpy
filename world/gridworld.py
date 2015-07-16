import matplotlib.pyplot as plt
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

    def plotModel(self, model):
        """ Product PDF files that show graphically the values of a model that
            is used to represent this world. This function does not know the meaning
            of the values stored by the model.
        """
        X = []
        Y = []
        V = [[] for i in range(self.nb_actions())]

        episode = Episode()

        for y in range(self.height):
            for x in range(self.width):
                Y.append(y)
                X.append(x)

                # Dummy episode that allows to fetch one value from the model
                episode.states.clear()
                episode.addState(self.encoding((x, y)))

                values = model.values(episode)

                for action, value in enumerate(values):
                    V[action].append(value)

        # Plot
        print('Plotting model')
        plt.figure()

        for a in range(self.nb_actions()):
            plt.figure()
            plt.scatter(X, Y, s=40, c=V[a])
            plt.colorbar()

            plt.savefig('model_%i.pdf' % a)
