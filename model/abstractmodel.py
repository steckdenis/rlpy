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

class AbstractModel(object):
    """ Model used to associate values with keys
    """

    def __init__(self, nb_actions):
        """ Constructor.

            @param nb_actions Number of possible actions in the world
        """
        self.nb_actions = nb_actions

    def learn(self, episodes):
        """ Update the model using the episodes provided. Each episode contains
            information about the states, actions and rewards visited by the agent.
        """
        raise NotImplementedError('The model does not implement learn()')

    def values(self, episode):
        """ Return the values associated with the last state of an episode
        """
        raise NotImplementedError('The model does not implement values()')

    def valuesForPlotting(self, episode):
        """ Return the values associated with the last state of an episode, possibly
            "faster" version used when plotting a model.
        """
        return self.values(episode)
