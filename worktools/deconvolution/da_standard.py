import pygaps as pg
import pygaps.parsing as pgp
import pygaps.characterisation as pgc
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, PchipInterpolator, interp1d
import numpy as np
import glob
from itertools import islice

Path = '/home/pcxtsbl/CodeProjects/labcore_upload/robert/aif/'
isotherm = pgp.isotherm_from_aif(
    f'{Path}ACC2700.aif',
)

unfiltered_results = {}

def list_split(
    List, min_length
):
    max_length = len(List)
    for i in range(min_length, max_length):
        for j in range(0, len(List), 1):
            split = (List[j:j + i])
            if len(split) > min_length:
                yield(split)


def split_isotherm(
    isotherm,
    p_limits: tuple[float, float] = [None, None],
    num_points = 10,
):
    pressures = list(isotherm.pressure())
    pressure = [x for x in pressures if x > p_limits[0]]
    pressure = [x for x in pressures if x < p_limits[1]]
    """
    loading = []
    for p in pressure:
        loading.append(float(isotherm.loading_at(p)))
        """
    split_pressures = list(list_split(
        pressure, num_points
    )
    )
    for split in split_pressures:
        split_loading = []
        for p in split:
            split_loading.append(float(isotherm.loading_at(p)))
        try:
            spline = interp1d(
                pgc.dr_da_plots.log_p_exp(list(split), 2), 
                pgc.dr_da_plots.log_v_adj(list(split_loading)),
                assume_sorted=False
            )
        except ValueError as e:
            print(e)
            continue
        plt.plot(split, spline(split))
        plt.show()
        second_derivative = spline.derivative(n=2)
        plt.plot(split, second_derivative(split))
        plt.show()


split_isotherm(isotherm, [1e-5, 0.1])
