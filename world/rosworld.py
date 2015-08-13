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

import threading

try:
    from queue import LifoQueue, Queue, Empty
except ImportError:
    from Queue import LifoQueue, Queue, Empty

try:
    import rospy
    import std_msgs
except:
    print('rospy (part of ROS) is not installed')

from .abstractworld import *
from .episode import *

class ROSProxy(object):
    """ RL-Glue agent that performs the actions it receives from rlpy
    """

    def __init__(self, subscriptions, publications):
        # Create queues for communication with the rlpy world
        self.observations_queue = LifoQueue(1)  # ROS updates constantly, keep only the last observation
        self.actions_queue = Queue()

        # Create the list of subscriptions so that the states can be
        # constructed from observations
        self.subscriptions = []
        self.last_state = []
        
        for subscription in subscriptions:
            self.last_state.append(0.0)
            sub = {}
            
            sub['index'] = len(self.subscriptions)  # Index in the state vector
            sub['subscriber'] = rospy.topics.Subscriber(
                subscription['path'],
                subscription['type'],
                self.subscription_callback,
                sub
            )
            sub['f'] = subscription.get('f', lambda x: x)
            
            self.subscriptions.append(sub)

        # Create the list of publications so that actions can be mapped to
        # publications
        self.publications = []
        self.actions = []
        
        for publication in publications:
            pub = {}
            
            pub['type'] = publication['type']
            pub['publisher'] = rospy.topics.Publisher(
                publication['path'],
                publication['type'],
                queue_size=10
            )

            # Add an action descriptor so that actions can be taken
            for value in publication['values']:
                self.actions.append((pub, value))
                
            self.publications.append(pub)

    def run(self):
        # Let ROS spin
        rospy.spin()
            
    def subscription_callback(self, data, sub):
        """ Called whenever something happens in the ROS world. This method
            updates the observation and publishes all the actions to be
            published
        """
        # Update the state. The last element of the state is the reward
        index = sub['index']
        f = sub['f']

        self.last_state[index] = float(f(data.data))

        self.observations_queue.put((
            tuple(self.last_state[:-1]),
            self.last_state[-1],
            False
        ))

        # Send the next action
        action = self.actions[self.actions_queue.get()]
        pub = action[0]
        value = action[1]

        pub['publisher'].publish(value)

    def numberOfActions(self):
        """ Number of possible actions, built based on the publications of this
            agent.
        """
        return len(self.actions) - 1    # The last action resets the world

    def observation(self):
        """ Wait for an observation to be available, and return a (state,
            reward, finished) tuple
        """
        return self.observations_queue.get()

    def setAction(self, action):
        """ Inform the agent that it has to take the given action
        """
        self.actions_queue.put(action)

class ROSWorld(AbstractWorld):
    """ Bridge between ROS (Robot OS) and this framework.

        This world subscribes to ROS topics and use them to produce states.
        When actions are performed, they are transformed to publishings in the
        ROS network.
    """

    def __init__(self, subscriptions, publications):
        """ Create and launch a new ROS agent
        
            @param subscriptions List of topics to which this world subscribes
                                 in order to construct its observations.
            @param publications List of topics on which this world publishes
                                when actions are performed.
                                
            Those two lists are lists of dictionaries, each dictionary
            containing those keys:
            
            - path : ROS path identifying the topic
            - type : message type (std_msgs.msg.Int32 for instance)
            - values : for publications, list of values that can be published
                       on this port. This allows rlpy to map action numbers
                       to ports and outputs.

            @note The last subscription gives the reward signal of the agent.
        """
        super(ROSWorld, self).__init__()

        # Initialize ROS (in the Python main thread)
        rospy.init_node(name='rlpy', anonymous=True, disable_signals=True)
        
        # Let the proxy initialize everything else
        self.proxy = ROSProxy(subscriptions, publications)
        self.thread = threading.Thread(target=(lambda: self.proxy.run()))

        # Start the thread
        print('Starting ROS thread...')

        self.thread.start()
        self.actions = self.proxy.numberOfActions()
        self.initial = self.proxy.observation()[0]   # First observation sent by ROS

        print('Started!')

    def nb_actions(self):
        return self.actions

    def reset(self):
        # Perform the last action (one that is not exposed through nb_actions)
        self.proxy.setAction(self.actions)

    def performAction(self, action):
        # Perform the action and wait for the next observation
        self.proxy.setAction(action)

        return self.proxy.observation()

