import pandas as pd
from dataclasses import dataclass, field


def closest(list, Number):
    aux = []
    for valor in list:
        aux.append(abs(Number-valor))
    return aux.index(min(aux))


def parse(file):
    psd_columns = ['w', 'V cum', 'dV/dw', 'S cum', 'dS/dw']
    isotherm_columns = ['P/P0', 'Amount Adsorbed', 'Fit']

    dat = pd.read_csv('./VRcP-650.CSV')
    dat.rename(columns={
        'Unnamed: 0': 'identifier',
        'Unnamed: 2': 'value',
    }, inplace=True
    )
    meta_dat = dat[
        ['identifier',
         'value']
    ].dropna().set_index('identifier').transpose().to_dict('records')[0]

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

    def porosity_slice(
        self,
        limits: tuple[float, float] = [None, None],
    ):
        if limits[0] > limits[1]:
            raise ValueError(
                f'your limits are in the wrong order; {limits}. '
                f'Please reverse'
            )
            return

        indeces = tuple[int, int]
        for l in limits:
            indeces[l] = closest(self.psd['w'], limits[l])
        V_range = (
            self.psd['Vcum'][indeces[0]], self.psd['Vcum'][indeces[1]]
        )
        S_range = (
            self.psd['Scum'][indeces[0]], self.psd['Scum'][indeces[1]]
        )
        V = V_range[1] - V_range[0]
        S = S_range[1] - S_range[0]

        return V, S


    def parse_saieus(self, file):
        with open(file) as f:
            reader = DataclassReader(f, saieus)
            reader.map('')
