# Reinforcement learning algorithms and experiments in Python

This repository contains Python code that can be used in order to experiment with reinforcement learning in Python. The code is organized in several components that can be mix and matched. For instance, different kinds of RL algorithms (Q-Learning, advantage, etc) can be tested on a specific world or problem. An algorithm can also be configured to use one of the possible models (Q-Learning can store the Q values in a simple dictionary, or using different kinds of function approximation methods).

* AbstractWorld: Environment and behavior of an agent. The world defines the number of possible actions, and produces observations and rewards when actions are carried out.
* AbstractLearning: Observes states and rewards and choose actions to perform.
* AbstractModel: Stores and retrieve values. For instance, a model is used to associate Q values to (state, action) pairs. A model can be discrete or based on function approximation.

# Dependencies

This project uses several machine-learning Python libraries. Most of them are optional, the program being able to run (with limited functionality) without them. Here is the list of dependencies, with instructions about how to install them.

* NumPy : Available on PyPi (`numpy`)
* Matplotlib : Available on PyPi (`matplotlib`)
* Theano (optional) : Available on PyPi (`Theano`)
* Keras (optional) : Available on PyPi (`Keras`)
* FANN2 (optional) : Available on PyPi (`fann2`)
* rlglue-py3 (optional) : https://github.com/steckdenis/rlglue-py3 . Python bindings for Python exist for some time but were never ported to Python3
