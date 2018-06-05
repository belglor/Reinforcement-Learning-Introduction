import numpy as np
import scipy.ndimage

from IPython.display import clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class RunningPlot():
    def __enter__(self):
        plt.clf()

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.draw()
        plt.pause(.1)
        return 0


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def reward_plotter(rewards, title, col, smooth_factor=0, include_sd=False):
    means = np.mean(rewards, 0)
    if smooth_factor >= 0:
        means = scipy.ndimage.filters.gaussian_filter1d(means, smooth_factor)
    plt.plot(means, col, label=title, alpha=0.75)
    if include_sd:
        sds = np.std(rewards, 0)
        if smooth_factor >= 0:
            sds = scipy.ndimage.filters.gaussian_filter1d(sds, smooth_factor)
        plt.plot(means + sds, col, alpha=0.1)
        plt.plot(means - sds, col, alpha=0.1)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()


def run_loop(env, agent, title, max_e=None, render=False):
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

        if render:
            with RunningPlot():
                plt.figure(1, figsize=(4, 4))
                plt.imshow(env.render())
                plt.title(title + ', step: {}'.format(i))
                clear_output(True)

        if d or i > 1e6:
            if since_last_plot > 1e4:
                with RunningPlot():
                    since_last_plot = 0
                    plt.figure(2, figsize=(8, 4))
                    plt.suptitle(title, x=0.1, y=1, fontsize=20, horizontalalignment='left')

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
                    clear_output(wait=True)

            ep_lens.append(i)
            rewards.append(r_sum)
            r_sum = 0; e += 1; i = 0
            s, r, d, _ = env.reset()

        if max_e and e >= max_e:
            break

    return ep_lens, rewards