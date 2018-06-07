
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
from agents import ApproximateNStepSARSA

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def run_loop(env, agent, title, max_e=None):
    t = 0; i = 0; e = 0
    s, r, d, _ = env.reset()
    a_ = agent.action(s)
    ep_lens = []; rewards = []
    r_sum = 0
    since_last_plot = 0

    while True:
        i += 1; t += 1; since_last_plot += 1
        a = a_
        s_, r, d, _ = env.step(a)
        a_ = agent.action(s_)

        agent.update(s=s, a=a, r=r, s_=s_, a_=a_, d=d)
        r_sum += r
        s = np.copy(s_)

        if (e + 1) % 100 == 0:
            env.render()
            

        if d:
           
            ep_lens.append(i)
            rewards.append(r_sum)
            r_sum = 0; e += 1; i = 0
            s, r, d, _ = env.reset()

        if max_e and e >= max_e:
            break

    return ep_lens, rewards


# In[4]:


num_runs = 1
eps_per_run = 1500
n = 10


# In[5]:


ApproxNSARSALearning_rewards = []
ApproxNSARSALearning_eplen = []
nbins = 8
ntiles = 8
env = TileEncoder(gym.make('MountainCar-v0'),nbins=nbins,ntiles=ntiles)
for i in range(num_runs):
    NSARSA_Learning = ApproximateNStepSARSA(env.obspace_shape(), env.nactions(), n=n)
    ep_lens, rewards = run_loop(env, NSARSA_Learning, 'NSARSALearning, n='+str(n), max_e=eps_per_run)
    ApproxNSARSALearning_rewards.append(rewards)
    ApproxNSARSALearning_eplen.append(ep_lens)

ApproxNSARSALearning_rewards  = np.array(ApproxNSARSALearning_rewards)
ApproxNSARSALearning_eplen = np.array(ApproxNSARSALearning_eplen)
env.close()

# In[6]:


plt.figure()
utils.reward_plotter(ApproxNSARSALearning_rewards, 'NSARSA', 'b')

axes = plt.gca()
axes.set_ylim([-200, 0])

plt.show()

#costToGo = NSARSA_Learning.costToGo()
    
res = 128
linspaces = []
for i in range(len(env.l_bound)): #do not consider number of tiles
    linspaces.append(np.linspace(env.l_bound[i],env.h_bound[i],num=res))  
mesh = np.meshgrid(linspaces[0],linspaces[1])

costToGo = np.zeros([res, res, env.nactions()])

for i in range(len(linspaces[0])):
    for j in range(len(linspaces[1])):
        s = env.encode([linspaces[0][i], linspaces[1][j]])
        qsa = NSARSA_Learning.linapprox(s)
        costToGo[i,j,:] = qsa
costToGo = -np.amax(costToGo,axis=2)

plt.imshow(costToGo)
plt.colorbar()