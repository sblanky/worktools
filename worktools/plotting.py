"""
Some functions for common plots I make in pyplot. Work in progress.
"""
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import pygaps.parsing as pgp


def normalize(
    x,
    newRange=(0, 1),
):
    r"""
    Simple function to normalize data within some range. Defaults to 0 to 1.
    Can be useful for normalizing TGA masses that go (slightly) negative"""

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
    r"""
    Define grid of subfigures according to groups of plots.
    Groups are defined by regular expressions, which correpond
    to filenames.
    """

    return plt.subplots(
        ncols=max_columns,
        nrows=len(regex_groups)/max_columns,
        constrained_layout=True,
    )


def tga(
    path: str = './',
    name_groups: list = ['*'],
    xlabel: str = 'T / $\\mathrm{^{\\circ}C}$', ylabel: str = 'wt.%',
    xlim: tuple = [0, 1000], ylim: tuple = [0, 100],
):
    """
    Plots set of subfigures of TGA data, grouped by regular expressions.
    """

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


def get_isotherms_psds(
    psd_path: str = './psd/',
    isotherm_path: str = './isotherms/'
):
    r"""
    Gets isotherms (from .aif or .csv) and pore size distributions 
    (psds, from .csv) then combines them into a dictionary of dataframes.
    This can then be used in plotting.
    """

    psd_files = glob.glob(f'{psd_path}*.CSV')
    psds = {}

    isotherm_files = glob.glob(f'{isotherm_path}*.csv')
    isotherm_files.append(
        glob.glob(f'{isotherm_path}*.aif')
    )
    isotherms = {}

    for f in psd_files:
        name = f.split(psd_path).split('*.CSV')
        psds[name] = pd.read_csv(
            f,
            usecols=['w', 'dV/dw', 'Vcum', 'dS/dw', 'Scum', ],
        )

    aif = re.findall(r'*.aif', isotherm_files)
    for f in aif:
        name = f.split(psd_path).split('*.aif')
        if name not in psds.keys():
            raise Exception(
                f'{name} found in {isotherm_path} but not in'
                f'{psd_path}. Not reading this isotherm. Please'
                f' check your data.'
            )

        else:
            isotherm = pgp.isotherm_from_aif(f)
            pressure = isotherm.data_raw['pressure']
            pressure_saturation = isotherm.data_raw['pressure_saturation']
            rel_pressure = pressure / pressure_saturation
            isotherms[name] = pd.DataFrame(
                list(zip(
                    rel_pressure, isotherm.data_raw['loading']
                )
                ),
                columns=['P/P0', 'loading', ],
            )

    csv = re.findall(r'*.csv', isotherm_files)
    for f in csv:
        name = f.split(psd_path).split('*.csv')
        if name not in psds.keys():
            raise Exception(
                f'{name} found in {isotherm_path} but not in'
                f'{psd_path}. Not reading this isotherm. Please'
                f' check your data.'
            )

        if name in isotherm.keys():
            raise Exception(
                f'Identically named .csv and .aif isotherm files'
                f' found.\n'
                f'name = {name}\n'
                f'Only using the .aif file. Please check your data.'
            )

        else:
            isotherm[name] = pd.read_csv(
                f,
                usecols=['relative pressure', 'loading', ]
            )
            isotherm[name].columns = ['P/P0', 'loading', ]

        isotherms_psds = {}
        for name in psds:
            isotherms_psds[name] = pd.concat(
                [isotherms[name], psds[name]],
                axis=1,
            )

        return isotherms_psds


def isotherms_psds_grouped(
    data,
    regex_groups: list = ['*', ],
    xlabel: str = '$P/P_0$',
    ylabel_left: str = '$Q\ /\ cm^3\ g^{-1}\ STP$',
    ylabel_right: str = '$PSD\ /\ cm^3\ g^{-1}\ \\unit{\angstrom}^{-1}$',
    psd_xlim: "tuple[float, float]" = [3.6, 20],
):
    r"""
    Plots groups of isotherms (defined by regex) alongside PSDs. Use
    get_isotherms_psds() to gather the data, then input it here.
    """

    fig, axs = plt.subplots(
        ncols=2, nrows=len(regex_groups),
        constrained_layout=True,
    )

    for i, g in enumerate(regex_groups):
        iso = axs[i, 0]
        psd = axs[i, 1]
        data_items = [key for key in data if re.match(g, key)]
        for dat in data_items:
            d = data[dat]
            iso.scatter(
                d['P/P0'], d['loading'],
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
        if i % 2 == 0:
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
