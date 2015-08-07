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

from world.abstractworld import *
from world.episode import *

class ModelWorld(AbstractWorld):
    """ World that returns next states and rewards based on a model that it learns
        from samples.
    """

    def __init__(self, makemodel, world):
        """ Create a new grid world.

            @param makemodel Function that creates a model. It takes a single
                             argument the number of "actions" for which this model
                             has to be configured.
            @param world "real" world to be approximated. This world will try to
                         look as close as possible to the real world. For instance,
                         it will encode its state in the same way.
        """
        super(ModelWorld, self).__init__()

        # Create a model
        self._model = makemodel(len(world.encoding(world.initial)) + 1)

        self.world = world
        self.reset()

    def nb_actions(self):
        return self.world.nb_actions()

    def reset(self):
        # Copy the initial state of the "real" world, so that random initial values
        # are handled correctly
        self.initial = self.world.encoding(self.world.initial)

        # Start a new episode with the initial state
        self._episode = Episode()
        self._state = self.initial

    def performAction(self, action):
        # Add the current state and the action to the episode
        self._episode.addState(self._make_state(self._state, action))
        self._episode.addAction(action)

        # Use this updated episode to predict the next state
        values = self._model.values(self._episode)
        state_update = values[:-1]
        reward = values[-1]

        # Update the current state
        old_state = self._episode.states[-1]
        new_state = tuple([a + b for a, b in zip(old_state, state_update)])

        self._episode.addReward(reward)
        self._episode.addValues(values)
        self._state = new_state

        return (new_state, reward, False)

    def performActionSupervised(self, action, target_state):
        self.performAction(action)

        # Force the target state
        self._state = target_state

    def learn(self, episodes):
        """ Use the episodes ((s, a, s', r) tuples) to train the model
        """
        model_episodes = []

        for episode in episodes:
            # Create an s -> s' episode based on the s -> Q episode received
            model_episode = Episode()

            for t in range(len(episode.states) - 1):
                state = episode.states[t]
                action = episode.actions[t]
                reward = episode.rewards[t]
                next_state = episode.states[t + 1]

                # State, action, reward
                model_episode.addState(self._make_state(state, action))
                model_episode.addAction(action)
                model_episode.addReward(reward)

                # The values to predict is the delta between state and next_state,
                # and the reward
                values = [b - a for a, b in zip(state, next_state)]
                values.append(reward)

                model_episode.addValues(values)

            model_episodes.append(model_episode)

        # Train the model on the model episodes
        self._model.learn(model_episodes)

    def _make_state(self, state, action):
        """ Make a "state" based on a real state and an action number
        """
        s = state[:]

        actions = [0.0 for a in range(self.nb_actions())]   # One variable per action
        actions[action] = 1.0                               # Set the variable corresponding to the action to 1

        return tuple(s) + tuple(actions)
