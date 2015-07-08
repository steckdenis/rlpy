import threading

try:
    from rlglue.agent.Agent import Agent
    from rlglue.agent import AgentLoader as AgentLoader
    from rlglue.types import Action
    from rlglue.types import Observation
except:
    print('RL-Glue (Python3 version) is not installed')

from .abstractworld import *
from .episode import *

class RLGlueAgent(Agent):
    """ RL-Glue agent that performs the actions it receives from rlpy
    """

    def __init__(self):
        self.lock = threading.Lock()

        # Condition used to wait for the agent to have been initialized
        self.init_condition = threading.Condition(self.lock)

        # Condition used to wait for actions from the rlpy world
        self.actions_condition = threading.Condition(self.lock)

        # Condition used to wait for observations from the rl-glue world
        self.observations_condition = threading.Condition(self.lock)

        self.initialzed = False
        self.min_action = 0
        self.max_action = 0
        self.nb_actions = 0

        self.last_state = None
        self.last_reward = None
        self.finished = False
        self.next_action = None

    def agent_init(self, taskSpec):
        with self.init_condition:
            # Parse the taskSpec in order to find the number of possible
            # actions.
            taskSpec = str(taskSpec)

            if 'ACTIONS INTS' not in taskSpec:
                raise Exception('Only discrete actions are supported by this framework')

            actions_ints = taskSpec.split('ACTIONS INTS')[1]                # VERSION ... ACTIONS INTS (0 3) REWARDS (0 10) ... -> (0 3) REWARDS (0 10) ...
            cleaned_up = actions_ints.replace('(', ' ').replace(')', ' ')   # (0 3) REWARDS (0 10) ... -> 0 3 REWARDS 0 10 ...
            parts = cleaned_up.split()[0:2]                                 # 0 3 REWARDS 0 10 -> ['0', '3']

            self.min_action = int(parts[0])
            self.max_action = int(parts[1])
            self.nb_actions = 1 + self.max_action - self.min_action
            self.initialized = True

            self.init_condition.notify_all()

    def agent_start(self, observation):
        return self.doStep(observation, 0.0)

    def agent_step(self, reward, observation):
        return self.doStep(observation, reward)

    def agent_end(self, reward):
        """ Place the agent in its "finished" state
        """

        with self.observations_condition:
            self.last_reward = reward
            self.finished = True

            self.observations_condition.notify_all()

    def agent_cleanup(self):
        pass

    def agent_message(self, inMessage):
        pass

    def waitForInitialized(self):
        """ Wait for the agent to be initialized and return the number of possible
            actions.
        """
        with self.init_condition:
            self.init_condition.wait_for(lambda: self.nb_actions > 0)

            return self.nb_actions

    def waitForObservation(self):
        """ Wait for an observation to be available, and return a (state, reward, finished)
            tuple
        """
        with self.observations_condition:
            self.observations_condition.wait_for(lambda: self.last_reward is not None)

            return (self.last_state, self.last_reward, self.finished)

    def setAction(self, action):
        """ Inform the agent that it has to take the given action
        """
        with self.actions_condition:
            # Set the next action and invalidate the last reward, so that the rlpy
            # world is forced to wait for the rl-glue world to update the observation
            self.next_action = action
            self.last_reward = None

            self.actions_condition.notify_all()

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
        with self.observations_condition:
            self.last_state = self.observationToState(observation)
            self.last_reward = reward
            self.finished = False

            self.observations_condition.notify_all()

        # Wait for instructions about the first action to perform
        with self.actions_condition:
            self.actions_condition.wait_for(lambda: self.next_action is not None)

            action = Action()
            action.intArray = [self.next_action]

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
        super().__init__()

        self.agent = RLGlueAgent()
        self.thread = threading.Thread(target=(lambda: _start_rlglue_agent(self.agent)))

        # Start the thread and wait for it to be initialized
        print('Starting RL-Glue thread...')

        self.thread.start()
        self.actions = self.agent.waitForInitialized()      # Number of actions
        self.initial = self.agent.waitForObservation()[0]   # And first observation sent by RL-Glue

        print('Started!')

    def nb_actions(self):
        return self.actions

    def reset(self):
        pass

    def performAction(self, action):
        # Perform the action and wait for the next observation
        self.agent.setAction(action)

        return self.agent.waitForObservation()

    def plotModel(self, model):
        print('An RL-Glue agent knows nothing about its world and cannot plot it, use an RL-Glue visualization tool')
