# Reinforcement learning algorithms and experiments in Python

This repository contains Python code that can be used in order to experiment with reinforcement learning in Python. The code is organized in several components that can be mix and matched. For instance, different kinds of RL algorithms (Q-Learning, advantage, etc) can be tested on a specific world or problem. An algorithm can also be configured to use one of the possible models (Q-Learning can store the Q values in a simple dictionary, or using different kinds of function approximation methods).

* AbstractWorld: Environment and behavior of an agent. The world defines the number of possible actions, and produces observations and rewards when actions are carried out.
* AbstractLearning: Observes states and rewards and choose actions to perform.
* AbstractModel: Stores and retrieve values. For instance, a model is used to associate Q values to (state, action) pairs. A model can be discrete or based on function approximation.
