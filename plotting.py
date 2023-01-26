
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np


def normalize(
    x,
    newRange=(0, 1),
):
    xmin, xmax = np.min(x), np.max(x)
    norm = (x - xmin)/(xmax - xmin)
    if newRange == (0, 1):
        return(norm)
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]


def subfig_definition(
    name_groups: list = ['*'],
    max_in_row: int = 2,
):
    return plt.subplots(
        len(name_groups), max_in_row,
        constrained_layout=True,
    )


def tga(
    path: str = './',
    name_groups: list = ['*'],
    xlabel: str = 'T / $\\mathrm{^{\\circ}C}$', ylabel: str = 'wt.%',
    xlim: tuple = [0, 1000], ylim: tuple = [0, 100],
):
    dat = {}

    path = './'
    files = glob.glob(f'{path}*.csv')

    for f in files:
        name = f.split(path)[1][:-4]
        dat[name] = pd.read_csv(f)
        if min(dat[name].Weight) < 0:
            dat[name].Weight = normalize(
                dat[name].Weight,
                (0, 100)
            )

    fig, axs = plt.subplots(
        len(name_groups)/2, 2,
        constrained_layout=True,
    )
    ax = axs.flat
    for i, g in enumerate(name_groups):
        ax = axs.flat[i]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        for key in dat:
            if re.match(g, key):
                d = dat[key]
                ax.plot(
                    d.Temperature, d.Weight,
                    label=key,
                )
        ax.legend(
            frameon=False,
        )

    return fig, axs
