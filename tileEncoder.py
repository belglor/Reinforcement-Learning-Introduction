# PACKAGES IMPORT
import numpy as np
import matplotlib.pyplot as plt
import time

class TileEncoder():
    def __init__(self, env, nbins=None, ntiles=None, l_bound=None, h_bound=None):        
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
        
    def encode(self, s, flatten=True):
        x = np.zeros(self.shapevec)
        shift_s = np.asarray(s) - np.asarray(self.l_bound)
        shift_upbound = np.asarray(self.h_bound) - np.asarray(self.l_bound)
        div = shift_upbound / (self.nbins-1)
        for i in range(self.ntiles):
# =============================================================================
             # UNIFORM OFFSET
#             offset = ( div / self.ntiles ) * i
#             op = (shift_s + offset) / div
#             idx = np.floor(op).astype(int)
#             x[idx[0],idx[1],i] = 1
# =============================================================================
            
#             # ASYMMETRICAL OFFSET
             offset  = ( div / self.ntiles ) * i
             offset_x = 1*offset[0]
             offset_y = 3*offset[1]
             asymmetric_s = [shift_s[0] - offset_x, shift_s[1] - offset_y]
             op = (asymmetric_s) / div
             idx = np.floor(op).astype(int)
             for j in range(len(idx)):
                 if idx[j] > self.ntiles-1:
                     idx[j] = self.ntiles -1           
                
             x[idx[0],idx[1],i] = 1
            
        if flatten:
            x = x.flatten()
            
        return x

    def step(self, action):
        s, r, d, i = self.env.step(action) 
        return self.encode(s), r, d, i
    
    def state(self):
        return self.encode(self.env.env.state)
    
    def reset(self, flatten=True):
        return self.encode(self.env.reset(), flatten), -1.0 , False, {}
    
    def obspace_shape(self):
        return self.shapevec
    
    def nactions(self):
        return self.env.action_space.n
    
    def render(self):
        return self.env.render()
    
    def close(self):        
        return self.env.close()
    
    def as_tiling(self, state, encode=True):
        x = state
        if encode:
            x = self.encode(state,False)
        else:
            x = np.reshape(x,[self.nbins, self.nbins, self.ntiles])
            
#        x = np.sum(x,axis=2)
#        plt.figure()
#        plt.imshow(x)
            
        tiled_rep = np.zeros([self.nbins*self.ntiles, self.nbins*self.ntiles, self.ntiles])
        for i in range(self.ntiles): #loop through tiling
            tile = x[:,:,i]
            idx = np.nonzero(tile)
            idx_x, idx_y = idx[0][0]*self.nbins, idx[1][0]*self.nbins
            start_x, end_x = idx_x+i, idx_x+self.ntiles+i
            # WITH UNIFORM OFFSET:
            #start_y, end_y = idx_y+i, idx_y+self.ntiles+i
            # WITH ASYMMETRICAL OFFSET: 
            start_y, end_y = idx_y+3*i, idx_y+self.ntiles+3*i
            
            if(start_x>=self.ntiles*self.nbins):
                start_x, end_x = self.ntiles-1, self.ntiles-1
            elif(end_x>=self.ntiles*self.nbins):
                end_x = self.ntiles-1
            elif(start_x>=self.ntiles*self.nbins):
                start_y, end_y = self.ntiles-1, self.ntiles-1
            elif(end_y>=self.ntiles*self.nbins):
                end_y = self.ntiles-1
                
            tiled_rep[start_x:end_x,start_y:end_y,i] = 1
            
        return tiled_rep
    
    def visualize_tiling(self, state, to_encode):
        tiling = self.as_tiling(state, to_encode)
        tiles = np.sum(tiling, axis=2)
        plt.figure()
        plt.imshow(tiles)