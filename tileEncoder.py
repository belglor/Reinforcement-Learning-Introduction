# PACKAGES IMPORT
import numpy as np

class TileEncoder():
    def __init__(self, env, nbins=None, ntiles=None, l_bound=None, h_bound=None):        
        env.reset()
        self.env = env
        
        if(nbins==None or ntiles==None):
            raise Exception
        self.nbins = nbins
        self.ntiles = ntiles
        
        self.shapevec = nbins * np.ones(len(env.env.state))
        self.shapevec = np.append(self.shapevec,ntiles).astype(int)
        
        inf_check = 100 #threshold for infinity-bounded obs_space 
        #No bounds specified -> use the one given by env
        if(h_bound == None):
            self.h_bound = env.observation_space.high
        if( any(i > inf_check for i in self.h_bound) ): #if obs space bounded to infinity, not tileable!
            print("Environment unbounded! Cannot tile code")
            raise Exception
                
        if(l_bound == None):
            self.l_bound = env.observation_space.low
        if( any(i < -inf_check for i in self.l_bound) ): #if obs space bounded to infinity, not tileable!
            print("Environment unbounded! Cannot tile code")
            raise Exception
            
        # Create grid-tiling through linspace
        self.gridtile = np.zeros([self.ntiles, env.observation_space.shape[0]])
        for i in range(env.observation_space.shape[0]):
            self.gridtile[:,i] = np.linspace(self.l_bound[i], self.h_bound[i], self.ntiles)
        
    def encode(self, s):
        
        x = np.zeros(self.shapevec)
        
        shift_s = np.asarray(s) - np.asarray(self.l_bound)
        shift_upbound = np.asarray(self.h_bound) - np.asarray(self.l_bound)
        
        for i in range(self.ntiles):
            div = shift_upbound / (self.nbins-1)
            offset = ( div / self.ntiles ) * i
            op = (shift_s + offset) / div
            idx = np.floor(op).astype(int)
            x[idx[0],idx[1],i] = 1
            
        return x.flatten()
    
    def checkgoal(self, s):
        if((np.abs(np.abs(s[0])-np.abs(self.env.env.goal_position))<0.001) or
            (s[0]>self.env.env.goal_position)):
            d = True
        else:
            d = False
        return d

    def step(self, action):
        s, r, d, i = self.env.step(action) 
        #d = self.checkgoal(s)
        return self.encode(s), r, d, i
    
    def state(self):
        return self.encode(self.env.env.state)
    
    def reset(self):
        return self.encode(self.env.reset()), -1.0 , False, {}
    
    def obspace_shape(self):
        return self.shapevec
    
    def nactions(self):
        return self.env.action_space.n
    
    def render(self):
        return self.env.render()
    
    def close(self):        
        return self.env.close()