import pandas as pd
from dataclasses import dataclass, field
import os
import warnings


def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number-valor))
    return aux.index(min(aux))


def parse(file):
    psd_columns = ['w', 'V cum', 'dV/dw', 'S cum', 'dS/dw']
    isotherm_columns = ['P/P0', 'Amount Adsorbed', 'Fit']

    dat = pd.read_csv(file)
    dat.rename(columns={
        'Unnamed: 0': 'identifier',
        'Unnamed: 2': 'value',
    }, inplace=True
    )
    meta_dat = dat[
        ['identifier',
         'value']
    ].dropna().set_index('identifier').transpose().to_dict('records')[0]
    name = os.path.split(file)[1]
    name = os.path.splitext(name)[0]
    if meta_dat['Sample (ID)'] != name:
        sampleid = meta_dat['Sample (ID)']
        warnings.warn(
            f'Name of file ({name}) does not match Sample ID '
            f'({sampleid}). Check your data'
        )

    psd_dat = dat[psd_columns].dropna().to_dict('list')
    isotherm_dat = dat[isotherm_columns].dropna().to_dict('list')

    return saieus(
        material=meta_dat['Sample (ID)'],
        model=meta_dat['Selected(Model1)'],
        lambda_regularisation=meta_dat['Lambda'],
        stdev=meta_dat['St Dev of Fit'],
        psd=psd_dat, isotherm=isotherm_dat,
    )


@dataclass
class saieus:
    material: str
    model: str
    lambda_regularisation: float
    stdev: float
    psd: dict = field(default_factory = lambda: {
        'w': None,
        'dV/dw': None,
        'V cum': None,
        'dS/dw': None,
        'S cum': None,
    })
    isotherm: dict = field(default_factory = lambda: {
        'P/P0': None,
        'Amount Adsorbed': None,
        'Fit': None,
    })

    def width_range(self):
        return (min(self.psd['w']), max(self.psd['w']))

    def peak(self):
        index_V = self.psd['dV/dw'].index(max(self.psd['dV/dw']))
        index_S = self.psd['dS/dw'].index(max(self.psd['dS/dw']))
        return self.psd['w'][index_V], self.psd['w'][index_S]

    def porosity_slice(
        self,
        limits: tuple[float, float] = [None, None],
    ):
        if limits[0] is None:
            limits[0] = 0
        if limits[1] is None:
            limits[1] = max(self.psd['w'])
        if limits[0] > limits[1]:
            raise ValueError(
                f'your limits are in the wrong order; {limits}. '
                f'Please reverse'
            )
            return

        indeces = []
        for i, l in enumerate(limits):
            indeces.append(closest(self.psd['w'], l))
        V_range = (
            self.psd['V cum'][indeces[0]], self.psd['V cum'][indeces[1]]
        )
        S_range = (
            self.psd['S cum'][indeces[0]], self.psd['S cum'][indeces[1]]
        )
        V = V_range[1] - V_range[0]
        S = S_range[1] - S_range[0]

        return V, S

    def pore_region_slice(
        self,
        region: str,
    ):
        if region is None:
            region = 'total'
        if region in ['micropore', 'micro']:
            return porosity_slice([0, 20])
        if region in ['mesopore', 'meso']:
            return porosity_slice([20, 500])
        if region is 'total':
            return(porosity_slice())
