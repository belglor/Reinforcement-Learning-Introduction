# PACKAGES IMPORT
import numpy as np
from math import floor, log
from itertools import zip_longest

# =============================================================================
#   SUTTON IDX HASHTABLE CODE
# =============================================================================
            
basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates
    
    
# =============================================================================
# SUTTON TILE ENCODING
# =============================================================================
    
class TileEncoder():
    def __init__(self, env, nbins=None, ntiles=None, l_bound=None, h_bound=None):        
        self.env = env
        self.iht = IHT(4096)
        
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

    def tiles (self, ihtORsize, numtilings, floats, ints=[], readonly=False):
        """returns num-tilings tile indices corresponding to the floats and ints"""
        qfloats = [floor(f*numtilings) for f in floats]
        Tiles = []
        for tiling in range(numtilings):
            tilingX2 = tiling*2
            coords = [tiling]
            b = tiling
            for q in qfloats:
                coords.append( (q + b) // numtilings )
                b += tilingX2
            coords.extend(ints)
            Tiles.append(hashcoords(coords, ihtORsize, readonly))
        return Tiles

    def tileswrap (self, ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
        """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
        qfloats = [floor(f*numtilings) for f in floats]
        Tiles = []
        for tiling in range(numtilings):
            tilingX2 = tiling*2
            coords = [tiling]
            b = tiling
            for q, width in zip_longest(qfloats, wrapwidths):
                c = (q + b%numtilings) // numtilings
                coords.append(c%width if width else c)
                b += tilingX2
            coords.extend(ints)
            Tiles.append(hashcoords(coords, ihtORsize, readonly))
        return Tiles
    
    def encode(self, s, a):
        x = np.zeros(4096)
        tiled = self.tiles(self.iht, 8, [(8*s[0]/(self.h_bound[0] - self.l_bound[0])), (8*s[1]/(self.h_bound[1] - self.l_bound[1]))] , [a])
        for i in range(tiled):
            x[tiled] = 1
            
        return x        
    
    def step(self, action):
        s, r, d, i = self.env.step(action) 
        x = self.encode(s,action)
        return x, r, d, i
    
    def state(self):
        return self.encode(self.env.env.state, 1)
    
    def reset(self):
        return self.encode(self.env.reset(),1), -1.0 , False, {}
    
    def obspace_shape(self):
        return self.shapevec
    
    def nactions(self):
        return self.env.action_space.n
    
    def render(self):
        return self.env.render()
    
    def close(self):        
        return self.env.close()