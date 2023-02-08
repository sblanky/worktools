import pandas as pd
import glob
import matplotlib.pyplot as plt
import pygaps as pg
import pygaps.parsing as pgp
import pygaps.characterisation as pgc
import os

Path = './'

class read_csv_isotherm:
    def __init__(self, **kwargs):
        self.temperature = kwargs.get(
            'temperature', 77
        ),
        self.pressure_mode = kwargs.get(
            'pressure_mode', 'relative'
        ),
        self.pressure_unit = kwargs.get(
            'pressure_unit', 'None'
        ),
        self.loading_basis = kwargs.get(
            'loading_basis', 'volume_gas'
        ),
        self.loading_unit=kwargs.get(
            'loading_unit', 'cm3'
        ),
        self.material_basis=kwargs.get(
            'material_basis', 'mass'
        ),
        self.material_unit=kwargs.get(
            'material_unit','g'
        ),


    def read_csv_isotherm(
        self,
        file,
        **kwargs,
    ):
        print(
            'reading csv, assuming N2 porosimetry isotherm '
            'i.e. 77 K, loading in cm3/g stp, pressure is '
            'relative. '
        )
        isotherm_data = pd.read_csv(file)
        try:
            relative_pressure = isotherm_data.relative_pressure
            loading = isotherm_data.loading
        except:
            raise ValueError(
                f'Columns don\'t adhere to headings needed to parse'
                f'csv file. {file_name} not read.'
            )
            break
        isotherm = pg.PointIsotherm(
            pressure=relative_pressure,
            loading=loading,

            material= os.path.splitext(file)[0],
            adsorbate=kwargs['adsorbate'],
            temperature=kwargs['temperature'],

            pressure_mode=kwargs['pressure_mode'],
            pressure_unit=kwargs['pressure_unit'],
            loading_basis=kwargs['loading_basis'],
            loading_unit=kwargs['loading_unit'],
            material_basis=kwargs['material_basis'],
            material_unit=kwargs['material_unit'],
        )
        return isotherm


def read_isotherm(file):
    file_name, file_extension = os.path.splitext(file)
    if file_extension == 'aif':
        return pgp.read_isotherm(file)
    if file_extension == 'csv':
        return read_csv_isotherm(file)
    else:
        raise ValueError(
            '{file} not read. Only csv and aif files parsed.'
        )
        pass


def analyse_isotherms(
    Path: str = './',
    outPath: str = './results/',
):
    outPaths = {}
    outPath['da'] = f'{outPath}/da/'

    for o in outPaths:
        if not os.path.exists(o):
            os.makedirs(o)

    for file in glob.glob(Path):
        isotherm = read_isotherm(file)
        da_results = pgc.dr_da_plots(
            isotherm, branch='ads',
            p_limits=[1e-5, 0.01],
            verbose=True
        )gt
        plt.savefig(
            {outPath['da']}+f'{isotherm.material}.png'
        )




