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

class AbstractLearning(object):
    """ Abstract class for learning strategies. Instances of this class receive
        state observations and rewards and have to choose the action to perform.
    """

    def __init__(self, nb_actions):
        """ Constructor.

            @param nb_actions Number of actions that are possible in the world
        """

        self.nb_actions = nb_actions

    def actions(self, episode):
        """ Return a probability density over the actions that should be performed
            given an history (represented by an Episode object)

            @return A tuple of two elements : a list of nb_actions elements,
                    each value giving the probability that the corresponding
                    action has to be taken, and the TD error.
        """
        raise NotImplementedError('The learning strategy does not implement action()')
