import glob
import os
import pygaps as pg
import pandas as pd


def csvtoaif(
    csv_file,
    in_dir: str = './csv/',
    out_dir: str = './aif/',
    **kwargs,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    isotherm = pd.read_csv(
        csv_file,
        names=['pressure', 'loading'],
        header=0,
    )
    name = csv_file.split(in_dir)[1][:-4]

    isotherm = pg.PointIsotherm(
        pressure=isotherm['pressure'],
        loading=isotherm['loading'],

        material=name,
        adsorbate=kwargs.get('adsorbate', 'N2'),
        temperature=kwargs.get('temperature', 77.0),

        temperature_unit=kwargs.get('temperature_unit', 'K'),
        pressure_unit=kwargs.get('pressure_unit', 'bar'),
        pressure_mode=kwargs.get('pressure_mode', 'relative'),
        loading_basis=kwargs.get('loading_basis', 'volume_gas'),
        loading_unit=kwargs.get('loading_unit', 'cm3'),
        material_basis=kwargs.get('material_basis', 'mass'),
        material_unit=kwargs.get('material_unit', 'g'),
    )

    isotherm.to_aif(f'{out_dir}{name}.aif')


def csvtoaif_dir(
    in_dir: str = './csv/',
    out_dir: str = './aif/',
    **kwargs,
):
    for f in glob.glob(f'{in_dir}*.csv'):
        csvtoaif(
            f,
            in_dir,
            out_dir,
            **kwargs,
        )
