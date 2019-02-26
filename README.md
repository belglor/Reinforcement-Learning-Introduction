## Reinforcement-Learning-Introduction
Code for simple reinforcement learning algorithms (Q-learning, SARSA) using tabular and linear approximation, as well as eligibility traces.

Three different jupyter notebooks are available. In each of them, a brief overview of the theory behind different RL methods is presented and accompanied by coding examples. 

This work is based on Richard S. Sutton and Andrew G. Barto. 1998. Introduction to Reinforcement Learning. MIT Press, Cambridge, MA, USA

## Table of contents

**[Tabular Methods][1]**
* Introduction to Reinforcement Learning: Tabular Methods
* The Algorithms
    - Temporal Difference Methods
    - SARSA
    - Q-learning
* Cliff Walking
* Experiments
* Results and Discussion
* N-step bootstrapping


**[Approximate Methods][2]**
* Linear Approximate Methods for Reinforcement Learning
    - (State / State-Action) Value Function Approximation
    - Stochastic Gradient Descent
    - Linear Approximation
* Algorithms for on-policy control with approximation
    - Episodic semi-gradient SARSA
    - The Deadly Triad
* Environment: OpenAI gym
    - Tile Coding
* Experiments
* Discussion


**[Eligibility Traces][3]**
* Eligibility Traces for Reinforcement Learning
* Generalizing between TD and Monte Carlo methods
    - TD($\lambda$)
    - Forward and Backward View
* True Online Methods
    - True Online TD($\lambda$)
    - SARSA($\lambda$)
* Experiments
* Discussion
  



[1]:https://github.com/belglor/Reinforcement-Learning-Introduction/blob/master/Tabular%20Methods.ipynb
[2]:https://github.com/belglor/Reinforcement-Learning-Introduction/blob/master/Approximate%20Methods.ipynb
[3]:https://github.com/belglor/Reinforcement-Learning-Introduction/blob/master/Eligibility%20Traces.ipynb
