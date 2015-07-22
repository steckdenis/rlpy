import threading

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

try:
    from rlglue.agent.Agent import Agent
    from rlglue.agent import AgentLoader as AgentLoader
    from rlglue.types import Action
    from rlglue.types import Observation

    baseclass = Agent
except:
    print('RL-Glue (Python3 version) is not installed')
    baseclass = object

from .abstractworld import *
from .episode import *

class RLGlueAgent(baseclass):
    """ RL-Glue agent that performs the actions it receives from rlpy
    """

    def __init__(self):
        # Create queues for communication with the rlpy world
        self.nb_actions_queue = Queue()
        self.observations_queue = Queue()
        self.actions_queue = Queue()

        self.last_state = None

    def agent_init(self, taskSpec):
        # Parse the taskSpec in order to find the number of possible actions.
        taskSpec = str(taskSpec)

        if 'ACTIONS INTS' not in taskSpec:
            raise Exception('Only discrete actions are supported by this framework')

        actions_ints = taskSpec.split('ACTIONS INTS')[1]                # VERSION ... ACTIONS INTS (0 3) REWARDS (0 10) ... -> (0 3) REWARDS (0 10) ...
        cleaned_up = actions_ints.replace('(', ' ').replace(')', ' ')   # (0 3) REWARDS (0 10) ... -> 0 3 REWARDS 0 10 ...
        parts = cleaned_up.split()[0:2]                                 # 0 3 REWARDS 0 10 -> ['0', '3']

        min_action = int(parts[0])
        max_action = int(parts[1])
        nb_actions = 1 + max_action - min_action

        # Put the number of actions in the nb_actions_queue so that the rlpy world
        # can detect that the number of actions is now known
        self.nb_actions_queue.put(nb_actions)

    def agent_start(self, observation):
        return self.doStep(observation, 0.0)

    def agent_step(self, reward, observation):
        return self.doStep(observation, reward)

    def agent_end(self, reward):
        """ Place the agent in its "finished" state
        """
        self.observations_queue.put((self.last_state, reward, True))

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        pass

    def numberOfActions(self):
        """ Wait for the agent to be initialized and return the number of possible
            actions.
        """
        return self.nb_actions_queue.get()

    def observation(self):
        """ Wait for an observation to be available, and return a (state, reward, finished)
            tuple
        """
        return self.observations_queue.get()

    def setAction(self, action):
        """ Inform the agent that it has to take the given action
        """
        self.actions_queue.put(action)

    def observationToState(self, observation):
        """ Transform an observation (sequence of integers, floats or chars) to
            a tuple that serves as state
        """
        if observation.intArray is not None and len(observation.intArray) > 0:
            return tuple(observation.intArray)
        elif observation.doubleArray is not None and len(observation.doubleArray) > 0:
            return tuple(observation.doubleArray)
        elif observation.charArray is not None and len(observation.charArray) > 0:
            return tuple(observation.charArray)
        else:
            return ()

    def doStep(self, observation, reward):
        """ Tell the rlpy world of an observation and a reward, and wait for it
            to return an action to take.

            @return An Action object that has to be transmitted to RL-Glue
        """

        # Inform the rlpy world of the observation
        self.last_state = self.observationToState(observation)

        self.observations_queue.put((self.last_state, reward, False))

        # Perform the next action
        action = Action()
        action.intArray = [self.actions_queue.get()]

        return action

def _start_rlglue_agent(agent):
    """ Start an RL-Glue agent and execute its event loop
    """
    AgentLoader.loadAgent(agent)

class RLGlueWorld(AbstractWorld):
    """ Bridge between RL-Glue and this framework.

        This world appears on the RL-Glue network as an agent, and is used by
        this framework as a world.
    """

    def __init__(self):
        """ Create and launch a new RL-Glue agent
        """
        super(RLGlueWorld, self).__init__()

        self.agent = RLGlueAgent()
        self.thread = threading.Thread(target=(lambda: _start_rlglue_agent(self.agent)))

        # Start the thread and wait for it to be initialized
        print('Starting RL-Glue thread...')

        self.thread.start()
        self.actions = self.agent.numberOfActions()
        self.initial = self.agent.observation()[0]                              # First observation sent by RL-Glue

        print('Started!')

    def nb_actions(self):
        return self.actions

    def reset(self):
        pass

    def performAction(self, action):
        # Perform the action and wait for the next observation
        self.agent.setAction(action)
        return self.agent.observation()

