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
from model.abstractmodel import *
from world.episode import *

from .modelworld import *

class TExploreModel(AbstractModel):
    """ Model based on TExplore (Hester, 2013), that learns a model of the world
        and use it to produce Q or Advantage values.
    """

    def __init__(self, world, makeworldmodel, model, learning, rollout_length):
        """ Initialize a new TExploreModel.

            @param world "real" world which will be approximated.
            @param makeworldmodel Function that creates models for ModelWorld,
                                  see ModelWorld.__init__ for details
            @param model Model used to store the Q or Advantage values discovered
                         by performing rollouts. This is the model that will predict
                         values in values()
            @param learning Learning algorithm used for the rollouts. Should be
                            the learning algorithm that will be used in the "real"
                            world, so that the values predicted by this model
                            match what the "real" learning algorithm expects.
            @param rollout_length Length of the rollouts performed by this model
        """
        super(TExploreModel, self).__init__(world.nb_actions())

        self._world = ModelWorld(makeworldmodel, world)
        self._model = model
        self._learning = learning
        self._rollout_length = rollout_length

    def values(self, episode):
        state = episode.states[-1]

        # Perform some rollouts from the current position
        num_rollouts = 3
        batch_size = 1

        self._world.run(self._model, self._learning, num_rollouts, self._rollout_length, batch_size, False, episode)

        # Use the model trained by the rollouts to predict the values
        values = self._model.values(episode)

        return values

    def valuesForPlotting(self, episode):
        return self._model.valuesForPlotting(episode)

    def learn(self, episodes):
        # Train the world on each of the episode.
        self._world.learn(episodes)
