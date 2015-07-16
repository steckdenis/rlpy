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
        
        for subscription in subscriptions:
            sub = {}
            
            sub['index'] = len(self.subscriptions)  # Index in the state vector
            sub['subscriber'] = rospy.topics.Subscriber(
                subscription['path'],
                subscription['type'],
                self.subscription_callback,
                sub
            )
            
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
            
        # Default dummy observation
        self.last_state = [0.0] * len(self.subscriptions)

    def run(self):
        # Let ROS spin
        rospy.spin()
            
    def subscription_callback(self, data, sub):
        """ Called whenever something happens in the ROS world. This method
            updates the observation and publishes all the actions to be
            published
        """
        # Update the state. The last element of the state is the reward
        self.last_state[sub['index']] = float(data.data)

        self.observations_queue.put((
            self.last_state[:-1],
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
        return len(self.actions)

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
        rospy.init_node(name='rlpy', anonymous=True)
        
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
        pass

    def performAction(self, action):
        # Perform the action and wait for the next observation
        self.proxy.setAction(action)

        return self.proxy.observation()

    def plotModel(self, model):
        print('A ROS agent knows nothing about its world and cannot plot it, use a ROS visualization tool')
