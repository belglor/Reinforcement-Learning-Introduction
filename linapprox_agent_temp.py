# PACKAGES IMPORT
import gym
import numpy as np

# ENVIRONMENT
class TileEncoder():
    def __init__(self, env, h_bound = None, l_bound = None, ntiles=None):        
        self.env = env
        
        if(ntiles==None):
            raise Exception
        self.ntiles = ntiles
        
        inf_check = 100 #threshold for infinity-bounded obs_space 
        #No bounds specified -> use the one given by env
        if(h_bound == None):
            self.h_bound = env.observation_space.high
        if( any(i > inf_check for i in self.h_bound) ): #if obs space bounded to infinity, not tileable!
            raise Exception
                
        if(l_bound == None):
            self.l_bound = env.observation_space.low
        if( any(i < -inf_check for i in self.l_bound) ): #if obs space bounded to infinity, not tileable!
            raise Exception
            
        # Create grid-tiling through linspace
        self.gridtile = np.zeros([self.ntiles, env.observation_space.shape[0]])
        for i in range(env.observation_space.shape[0]):
            self.gridtile[:,i] = np.linspace(self.l_bound[i], self.h_bound[i], self.ntiles)
        
    def encode(self, s):
        x = np.zeros([self.ntiles, self.env.observation_space.shape[0]])
        
        mask = s < self.gridtile #Create mask
        for i in range(self.env.observation_space.shape[0]):
            index = [j for j, k in enumerate(mask[:,i]) if k == False][-1]
            x[index,i] = 1
        return np.concatenate(x.T)  
        
    def step(self, action):
        s, r, d, i = self.env.step(action)        
        return self.encode(s), r, d, i
    
    def state(self):
        return self.encode(self.env.env.state)
    
    def reset(self):
        return self.env.reset()
    
    def obspace_shape(self):
        return self.gridtile.shape
    
    def nactions(self):
        return self.env.action_space.n
    
    def render(self):
        return self.env.render()
    
    def close(self):        
        return self.env.close()
    
    
# AGENT
class agent():
    
    def __init__(self, obspace_shape, nactions): 
        
        self.w = np.zeros([obspace_shape[0]*obspace_shape[1],nactions])
        
        shape_vec = []
        for i in range(obspace_shape[1]):
            shape_vec.append(obspace_shape[0])
        shape_vec.append(nactions)  
        
        #stored variables for n-step SARSA
        self.t = 0 
        self.T = np.inf
        self.n = 10
        self.alpha = 0.001
        self.e = 0.1 
        self.gamma = 1 - 0.001
        
        #variables trackers
        self.sarsatrack = []
        self.gtracker = []
        
    def linapprox(self, s_t, a_t):
        w = self.w[:,a_t]
        qsa = w.T.dot(s_t)
        return qsa
    
    def action(self, x_t): #Take as input the current state (ENCODED), returns e-greedy action
        possible_acts = self.w.T.dot(x_t)
        best_act_val, best_act = max(possible_acts), np.argmax(possible_acts)
        if(np.random.random()>self.e):
            best_act = np.random.randint(0,len(possible_acts))
        return best_act
     
    #TODO: DONT USE QSA, it depends solely on the encoded action and weights!
    #TODO: make updates so they linearly approximate QSA (make linapprox method)
    def update(self, s_t, a_t, r_t1, s_t1, d_t1): 
        # s     = initial state
        # a     = e-greedy action
        # r     = reward after performing action
        # s_1   = subseguent state after taking a
        # d     = done flag  
        self.t = self.t + 1
        
        if(self.t < self.T):
            if(d_t1):
                self.T = self.t + 1 
                a_t1 = 0
            else:
                a_t1 = self.action(s_t1)                
            self.sarsatrack.append([s_t,a_t,r_t1,s_t1,a_t1])
        G = 0  
        tau = self.t - self.n + 1
        if(tau>=0):          
            #for i in range(min([tau+self.n,self.T])):
            for i in range(min([self.n,self.T-tau])):
                G = G + (self.gamma**i)*(self.sarsatrack[tau+i-1][2]) #tau+i accesses timestep, 2 accesses the reward
                            
            if(tau+self.n < self.T):
                s_tau_n = self.sarsatrack[tau+self.n-2][0]
                a_tau_n = self.sarsatrack[tau+self.n-2][1]
                qsa = agent.linapprox(s_tau_n,a_tau_n)
                G = G + (self.gamma**self.n)*qsa
                
            s_tau   = self.sarsatrack[tau-1][0]
            a_tau   = self.sarsatrack[tau-1][1]   
            qsa = agent.linapprox(s_tau,a_tau)
            self.w[:,a_tau] = self.w[:,a_tau] + self.alpha*(G - qsa)*s_tau
        
        self.gtracker.append(G)
        
        return (tau == self.T - 1)
    
    def reset(self):
        tmp_G = self.gtracker.copy()
        tmp_sarsatrack = self.sarsatrack.copy()
        self.gtracker = []
        self.sarsatrack = []
        self.T = np.inf
        self.t = 0        
        return tmp_G, tmp_sarsatrack
        
        
        
if __name__=='__main__':
    ntiles = 64
    env = TileEncoder(gym.make('MountainCar-v0'),ntiles=ntiles)
    
    agent = agent(env.obspace_shape(), env.nactions())
    
    max_events = 1000
    #Store initial state
    
    
    for i in range(max_events):
        print("###################################")
        print("     NEW EVENT STARTED: "+ str(i) + "/" + str(max_events))
        print("###################################")
        env.reset()
        agent.reset()
        s_t = env.state()
        #GENERATE EVENT
        isdone = False
        while(not isdone):
            #env.render()
            if(agent.t < agent.T):
                #Have s_t, call agent and compute next action
                a_t = agent.action(s_t)
                
                #Having a_t, call env and return r_t1, s_t1, d    
                s_t1, r_t1, d_t1, info = env.step(a_t)
                d_t1 = (np.abs(np.abs(env.env.env.state[0])-np.abs(env.env.env.goal_position))<0.0001)
                #print(env.env.env.state[0])
                #print(env.env.env.goal_position)
                if(env.env.env.state[0]==0.6):
                    break
            #Perform update
            isdone = agent.update(s_t,a_t,r_t1,s_t1,d_t1)
            
            #Update parameters
            s_t = s_t1
        print(agent.t)

    qsa = np.zeros([ntiles,ntiles,env.nactions()])
    for i in range(env.nactions()):
        for j in range(ntiles):
            for k in range(ntiles):
                s_t = np.zeros(2*ntiles)
                s_t[j] = 1
                s_t[ntiles+k] = 1
                qsa[j,k,i] = agent.linapprox(s_t,i)
    
    costtogo = -np.amax(qsa,axis=2)
    
    import matplotlib.pyplot as plt
    plt.imshow(costtogo)
    
    
#    #%%
#    low = env.env.observation_space.low
#    high = env.env.observation_space.high
#    x = np.linspace(low[0],high[0])
#    y = np.linspace(low[1],high[1])
#    x = (x*1000).astype(int)
#    y = (y*1000.astype(int)
#
#    from RLtoolkit.Tiles import tiles
#    
#    tiles.tiles(8,64,[x,y])