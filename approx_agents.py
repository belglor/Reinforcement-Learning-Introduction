from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ApproximateNStepQLearning:
    def __init__(self, state_shape, num_actions, n=1):
        self.num_actions = num_actions
        self.tab_shape = np.hstack([state_shape, num_actions])
        self.w = np.zeros(np.hstack([np.prod(np.array(state_shape)), num_actions]))
        self.n = n

        self.min_eps = 0.1
        self.decay_len = 1 # 1e4
        self.alpha = 0.1 / n
        self.gamma = 0.99
        self.t = 0
        self.exp = []
        # self.Qtable = np.zeros(self.tab_shape) #TODO: is Q-table needed?

    @property
    def eps(self):
        _eps = 1 - self.t/self.decay_len
        return np.maximum(self.min_eps, _eps)
    
    def linapprox(self, s, a=None):
        qsa = self.w.T.dot(s)
        if(not a==None):
            qsa = qsa[a]
        return qsa

    def action(self, state):
        self.t += 1
        if np.random.uniform() > self.eps:
            a = np.argmax(self.linapprox(state))
        else:
            a = np.random.randint(self.num_actions)
        return a

    def compute_G(self):
        """ Discounted reward"""
        G = 0
        for i in range(len(self.exp)):
            G = G + (self.gamma ** i) * self.exp[i][2]
        return G

class ApproximateNStepSARSA(ApproximateNStepQLearning):
    def update(self, s, a, r, s_, a_, d, **kwargs):
        self.exp.append([s, a, r, s_, a_])          #NOTE: s_, a_ are tau+n (stacked at the end of exp stack)
        
        if d:  # Done --> loop through experience
            while len(self.exp) > 0:
                fs, fa, fr, fs_, fa_ = self.exp[0]  #NOTE: fs, fa are tau (taken at beginning of exp stack)
                G = self.compute_G()
                self.exp.pop(0)
                
                qsa = self.linapprox(fs,fa)
                G += (self.gamma ** self.n) * qsa
                
                self.w[:,fa] +=  self.alpha*(G - qsa)*fs
            
        elif len(self.exp) <= self.n:  # to early -- move along
            pass
        else:  # Normal n-step update w. bootstrapping
            fs, fa, fr, fs_, fa_ = self.exp[0]
            G = self.compute_G()
            self.exp.pop(0)

            qsa = self.linapprox(fs,fa)
                
            self.w[:,fa] +=  self.alpha*(G - qsa)*fs        
