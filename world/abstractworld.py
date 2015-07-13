from numpy.random import choice

from .episode import *

def encode_identity(state):
    """ Identity encoding, does not change the state
    """
    return state

def _encode_onehot(state, ranges):
    """ Encode discrete state variables to a sequence of floats that each have
        a value of 0 or 1. For instance, (2, 3) is encoded as (0, 0, 1, 0, ...,
        0, 0, 0, 1, 0, ...).

        @param state State to encode
        @param ranges Ranges of the variables in the state (width and height of
                      a grid for instance)
    """
    res = [0.0] * sum(ranges)
    offset = 0

    for index, value in enumerate(state):
        res[offset + int(value)] = 1.0
        offset += ranges[index]

    return tuple(res)

def make_encode_onehot(ranges):
    """ Return an encode function that encodes states using the provided ranges
    """
    return lambda s: _encode_onehot(s, ranges)

class AbstractWorld(object):
    """ Abstract world that receives actions and produces new states
    """
    def __init__(self):
        self.encoding = encode_identity

    def nb_actions(self):
        """ Return the number of actions that can be performed. This number
            cannot change during the lifetime of the world.
        """
        raise NotImplementedError('The world does not implement nb_actions()')

    def reset(self):
        """ Reset the world in its original configuration, as if no agent performed
            actions on it.
        """
        raise NotImplementedError('The world does not implement reset()')

    def performAction(self, action):
        """ Perform an action on the world and return a tuple (state, reward, finished)

            @note The state returned must not be encoded. AbstractWorld takes care
                  of the encoding when needed.
        """
        raise NotImplementedError('The world does not implement performAction()')

    def plotModel(self, model):
        """ Produce PDF files representing the given model in the current world.
        """
        print('No plot produced for this model as it does not reimplement plotModel()')

    def run(self, model, learning, num_episodes, max_episode_length, batch_size):
        """ Simulate an agent in this world.

            @param learning Learning algorithm used by the agent
            @param model Model that will be used for learning
            @param num_episodes Number of episodes that are simulated
            @param max_episode_length Maximum number of steps allowed per episode
            @param batch_size Off-policy learning happens once after every @p batch_size episodes

            @return A list of Episode objects
        """
        episodes = []
        learn_episodes = []
        possible_actions = list(range(self.nb_actions()))

        try:
            for e in range(num_episodes):
                episode = Episode()

                # Initial state
                episode.addState(self.encoding(self.initial))
                episode.addValues(model.values(episode))
                self.reset()

                finished = False
                steps = 0

                # Perform the steps
                while steps < max_episode_length and not finished:
                    probas = learning.actions(episode)
                    action = choice(possible_actions, p=probas)

                    state, reward, finished = self.performAction(action)

                    episode.addReward(reward)
                    episode.addAction(action)
                    episode.addState(self.encoding(state))
                    episode.addValues(model.values(episode))

                    steps += 1

                # Let the learning update the Q-value of the last state visited
                learning.actions(episode)

                # If a batch has been finished, learn
                episodes.append(episode)
                learn_episodes.append(episode)

                print(e, episode.cumulative_reward)

                if len(learn_episodes) == batch_size:
                    model.learn(learn_episodes)
                    learn_episodes.clear()
        except KeyboardInterrupt:
            # Allow the user to gracefully interrupt the learning process
            pass

        return episodes