import numpy as np
import scipy.ndimage

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