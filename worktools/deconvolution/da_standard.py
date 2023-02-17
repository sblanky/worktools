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
    num_points: int = 10,
):
    pressure_full = list(isotherm.pressure())
    pressure = [x for x in pressure_full if x > p_limits[0]]
    pressure = [x for x in pressure if x < p_limits[1]]
    pressure_lists = list(list_split(
        pressure, num_points
    )
    )
    linear_pressure_lists = []
    for pressure_list in pressure_lists:
        loading_list = []
        for p in pressure_list:
            loading_list.append(float(isotherm.loading_at(p)))
        log_p_exp = np.flip(
            pgc.dr_da_plots.log_p_exp(
                list(pressure_list), 2
            )
        )
        log_v_adj = np.flip(
            pgc.dr_da_plots.log_v_adj(
                list(loading_list)
            )
        )
        try:
            spline = PchipInterpolator(
                log_p_exp, log_v_adj,
            )
        except ValueError as e:
            print(e)
            continue
        second_derivative = spline.derivative(nu=2)

        second_derivative_range = abs(
            second_derivative(min(log_p_exp)) - second_derivative(max(log_p_exp))
        )
        #print(second_derivative_range)

        if second_derivative_range < 1e-3:
            linear_pressure_lists.append(pressure_list)

    return linear_pressure_lists


def isotherm_splitter(
    original_isotherm = None,
    pressure: list = None,
):
    loading = list() 
    for p in pressure:
        loading.append(
            original_isotherm.loading_at(p)
        )

    return pg.PointIsotherm(
        pressure=pressure,
        loading=loading,

        material=isotherm.material,
        adsorbate=original_isotherm.adsorbate,
        temperature=original_isotherm.temperature,

        pressure_mode=original_isotherm.pressure_mode,
        pressure_unit=original_isotherm.pressure_unit,
        loading_basis=original_isotherm.loading_basis,
        loading_unit=original_isotherm.loading_unit,
        material_basis=original_isotherm.material_basis,
        material_unit=original_isotherm.material_unit,
    )


def dr_plotter(
    linear_pressure_lists: list,
    isotherm,
):
    results_dict = {}
    n=0
    for pressure_list in linear_pressure_lists:
        results = pgc.dr_da_plots.dr_plot(
            isotherm,
            p_limits=[min(pressure_list), max(pressure_list)],
        )
        results['length'] = len(pressure_list)
        results['pressure'] = pressure_list
        results_dict[n] = results
        n+=1

    return results_dict


def dr_results_sorter(
    results_dict
):
    results_filtered = {}
    for r in results_dict:
        result = results_dict[r]
        if abs(result['corr_coef']) > 0.999:
            results_filtered[r] = results_dict[r]

    return results_filtered


linear_pressure_lists = split_isotherm(
    isotherm, p_limits=[1e-5, 0.1]
)
print(linear_pressure_lists)
results_dict = dr_plotter(linear_pressure_lists, isotherm)
filtered_results = dr_results_sorter(results_dict)
for f in filtered_results:
    filt = filtered_results[f]
    print(filt['pressure'])
    pgc.dr_da_plots.dr_plot(
        isotherm,
        p_limits=[min(filt['pressure']), max(filt['pressure'])],
        verbose=True,
    )
    plt.show()
