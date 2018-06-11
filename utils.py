import numpy as np
import scipy.ndimage

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class RunningPlot():
    def __enter__(self):
        plt.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        #plt.draw()
        plt.pause(.1)
        return 0


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def reward_plotter(rewards, title, col, smooth_factor=2, include_sd=False):
    means = np.mean(rewards, 0)
    means = scipy.ndimage.filters.gaussian_filter1d(means, smooth_factor)
    plt.plot(means, col, label=title, alpha=0.75)
    if include_sd:
        sds = np.std(rewards, 0)
        sds = scipy.ndimage.filters.gaussian_filter1d(sds, smooth_factor)
        plt.plot(means + sds, col, alpha=0.1)
        plt.plot(means - sds, col, alpha=0.1)
    plt.legend()
    
def costToGo(res, env, agent):
    linspaces = []
    for i in range(len(env.l_bound)): #do not consider number of tiles
        linspaces.append(np.linspace(env.l_bound[i],env.h_bound[i],num=res))  
                    
    costToGo = np.zeros([res, res, env.nactions()])

    for i in range(len(linspaces[0])):
        for j in range(len(linspaces[1])):
            s = env.encode([linspaces[0][i], linspaces[1][j]])
            qsa = agent.linapprox(s)
            costToGo[i,j,:] = qsa
    costToGo = -np.amax(costToGo,axis=2)
    return costToGo

#TODO: unify this with utils.run_loop so it is accomodated in both

def approx_run_loop(env, agent, title, max_e=None):
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
            if (e + 1) % 100 == 0:
                with RunningPlot():
                    plt.figure(1, figsize=(4, 4))
                    plt.imshow(costToGo(128, env, agent))
                    plt.title(title + ', episode: {}'.format(e) + ', step: {}'.format(i))
                    plt.colorbar()
        
                
            ep_lens.append(i)
            rewards.append(r_sum)
            r_sum = 0; e += 1; i = 0
            s, r, d, _ = env.reset()

        if max_e and e >= max_e:
            break

    return ep_lens, rewards