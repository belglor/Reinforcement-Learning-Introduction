import numpy as np
import scipy.ndimage

from IPython.display import clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



class RunningPlot:
    def __init__(self, t=1e-6):
        self.t = t

    def __enter__(self):
        # TODO: add a notebook check - clf vs clear_output
        # plt.clf()
        clear_output(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.draw()
        plt.pause(self.t)


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def reward_plotter(rewards, title, col='b', smooth_factor=0, include_sd=False):
    means = np.mean(rewards, 0)
    if smooth_factor >= 0:
        try:
            means = scipy.ndimage.filters.gaussian_filter1d(means, smooth_factor)
        except ZeroDivisionError:
            pass
    plt.plot(means, col, label=title, alpha=0.75)
    if include_sd:
        sds = np.std(rewards, 0)
        if smooth_factor >= 0:
            try:
                sds = scipy.ndimage.filters.gaussian_filter1d(sds, smooth_factor)
            except ZeroDivisionError:
                pass
        plt.plot(means + sds, col, alpha=0.1)
        plt.plot(means - sds, col, alpha=0.1)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()


def run_loop(env, agent, title, max_e=None, render=False, update=True, plot_frequency=5e3):
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

        if update:
            agent.update(s=s, a=a, r=r, s_=s_, a_=a_, d=d)
        r_sum += r
        s = np.copy(s_)

        if render:
            with RunningPlot(0.1):
                plt.figure(1, figsize=(4, 4))
                plt.imshow(env.render())
                plt.title(title + ', step: {}'.format(i))

        if d or i > 1e6:
            if since_last_plot > plot_frequency:
                with RunningPlot():
                    since_last_plot = 0
                    plt.figure(2, figsize=(8, 4))
                    plt.suptitle(title + ', episode: '+str(e), x=0.1, y=1, fontsize=20, horizontalalignment='left')

                    plt.subplot(121)
                    plt.title('Highest action value')
                    img1 = plt.imshow(np.max(agent.Qtable, -1))
                    plt.axis('equal', frameon=True)
                    colorbar(img1)

                    plt.subplot(122)
                    plt.title('Movement Heatmap')
                    img2 = plt.imshow(env.heat_map)
                    plt.axis('equal')
                    colorbar(img2)

            ep_lens.append(i)
            rewards.append(r_sum)
            r_sum = 0; e += 1; i = 0
            s, r, d, _ = env.reset()
            a_ = agent.action(s)

        if max_e and e >= max_e:
            break

    return ep_lens, rewards


def costToGo(res, env, agent):
    linspaces = []
    for i in range(len(env.l_bound)):  # do not consider number of tiles
        linspaces.append(np.linspace(env.l_bound[i], env.h_bound[i], num=res))

    costToGo = np.zeros([res, res, env.nactions()])

    for i in range(len(linspaces[0])):
        for j in range(len(linspaces[1])):
            s = env.encode([linspaces[0][i], linspaces[1][j]])
            qsa = agent.linapprox(s)
            costToGo[i, j, :] = qsa
    costToGo = -np.amax(costToGo, axis=2)
    return costToGo


def approx_run_loop(env, agent, title, max_e=None, render=False,):
    t = 0; i = 0; e = 0
    s, r, d, _ = env.reset()   
    a_ = agent.action(s)
    ep_lens = []; rewards = []
    r_sum = 0
    since_last_plot = 0

    while True:
        if d: #NOTE: resetting here should cause no issues
            s, r, d, _ = env.reset()
        i += 1; t += 1; since_last_plot += 1
        a = a_
        s_, r, d, _ = env.step(a)
        a_ = agent.action(s_)

        agent.update(s=s, a=a, r=r, s_=s_, a_=a_, d=d)
        r_sum += r
        s = np.copy(s_)

        if render and (e + 1) % 500 == 0:
            env.render()

        if d:
            if (e + 1) % 100 == 0:
                with RunningPlot():
                    plt.figure(1, figsize=(4, 4))
                    img = plt.imshow(costToGo(128, env, agent))
                    plt.title(title + ', episode: {}'.format(e))
                    colorbar(img)
                    #env.visualize_tiling(s,False)

            ep_lens.append(i)
            rewards.append(r_sum)
            r_sum = 0; e += 1; i = 0
            #s, r, d, _ = env.reset() 
            #NOTE: resetting here cancels the info of the last state visited after termination
            #Might be SOMEHOW detrimental. Better 
                        

        if max_e and e >= max_e:
            break

    return ep_lens, rewards

