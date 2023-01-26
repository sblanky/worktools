
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
    regex_groups: list = ['*'],
    max_columns: int = 2,
):
    return plt.subplots(
        ncols=max_columns,
        nrows=len(regex__groups)/max_columns,
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


def isotherms_psds_grouped(
    data,
    regex_groups: list = ['*',],
    xlabel: str = '$P/P_0$',
    ylabel_left: str = '$Q\ /\ cm^3\ g^{-1}\ STP$', 
    ylabel_right: str = '$PSD\ /\ cm^3\ g^{-1}\ \\unit{\angstrom}^{-1}$',
    psd_xlim: list = [3.6, 200]
):

    fig, axs = plt.subplots(
        ncols=2, nrows=len(regex_groups),
        constrained_layout=True,
    )

    for i, g in enumerate(regex_groups):
        iso = axs[i,0]
        psd = axs[i,1]
        dat = [key for key in dat if re.match(g, key)]
        for d in dat:
            iso.scatter(
                d['P/P0']. d['loading'],
                clip_on=False,
                marker='^',
            )
            psd.plot(
                d['w'], d['dV/dw'],
                label=d,
            )
        psd.legend(frameon=False)

    for i, ax in enumerate(axs.flat):
        ax.set_ylim(0, ax.get_ylim()[1])
        if i%2 == 0:
            ax.set_ylabel(ylabel_left)
            ax.set_xlim(0, 1)

        else:
            ax.set_ylabel(ylabel_right)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.set_xlim(psd_xlim[0], psd_xlim[1])

        if i in [len(axs), len(axs)-1]:
            ax.set_xlabel(xlabel)

    return fig, axs
