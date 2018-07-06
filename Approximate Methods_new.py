
# coding: utf-8

# ### TODO
# * Explain: Approximations
# * Explain: Linear approx.
# * Explain: Q-learning extension
# * Explain: SARSA extension
# * Explain: Mountain Car env
# * code: Run and compare (on mountain car)
# 
# 
# ### DONE
# 
# 
# ### NOTES

# # Linear Approximate Methods for Reinforcement Learning
# 

# In[2]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import gym

import utils
from tileEncoder import TileEncoder
from agents import ApproximateNStepSARSA, RandomAgent

get_ipython().run_line_magic('matplotlib', 'inline')

## Run settings
num_runs = 10 # Number of runs to average rewards over
eps_per_run = 1000 # Number of episodes (terminations) per run

nbins = 8
ntiles = 8

# n parameter in n-step Bootstrapping
n1 = 8 # agent 1
n2 = 20  # agent 2


# In[4]:


ApproxNSARSALearning_rewards_n1 = []
env = TileEncoder(gym.make('MountainCar-v0'),nbins=nbins,ntiles=ntiles)
for i in range(num_runs):
    NSARSA_Learning = ApproximateNStepSARSA(env.obspace_shape(), env.nactions(), n=n1)
    _, rewards = utils.approx_run_loop(env, NSARSA_Learning, str(i)+': NSARSA, n='+str(n1), max_e=eps_per_run, render=True)
    ApproxNSARSALearning_rewards_n1.append(rewards)
    
env.close()

ApproxNSARSALearning_rewards_n1  = np.array(ApproxNSARSALearning_rewards_n1)

# In[5]:

plt.figure()
utils.reward_plotter(ApproxNSARSALearning_rewards_n1, 'NSARSA, n='+str(n1), col='b', smooth_factor=1, include_sd=False)
axes = plt.gca()
axes.set_ylim([-225, 0])
plt.show()