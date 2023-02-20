import pygaps as pg
import pygaps.parsing as pgp
import pygaps.characterisation as pgc
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy as np

from pygaps.utilities.exceptions import CalculationError
from pygaps.utilities.exceptions import ParameterError


def list_split(
    List, min_length
):
    max_length = len(List)
    for i in range(min_length, max_length):
        for j in range(0, len(List), 1):
            split = (List[j:j + i])
            if len(split) > min_length:
                yield(split)


def curvature_filter(
    x: list,
    y: list,
    tolerance: float = 0.1,
):
    try:
        spline = PchipInterpolator(x, y)
    except ValueError as e:
        print(e)
        return False

    second_derivative = spline.derivative(nu=2)

    curvature = abs(max(second_derivative(x)) - min(second_derivative(x)))
    if curvature > tolerance:
        return False
    else:
        return True


def volume_filter(
    micropore_capacity: float,
    micropore_capacity_limit: float,
):
    if micropore_capacity < micropore_capacity_limit:
        return True


def correlation_filter(
    corr_coef: float,
    min_corr_coef: float = 0.99,
):
    if abs(corr_coef) > min_corr_coef:
        return True


def split_pressures(
    isotherm,
    p_limits: tuple[float, float] = [None, None],
    num_points: int = 10,
):
    pressure_full = list(isotherm.pressure())
    pressure = [x for x in pressure_full if x > p_limits[0]]
    pressure = [x for x in pressure if x < p_limits[1]]

    return list(list_split(
        pressure, num_points
    )
    )


def isotherm_from_pressures(
    original_isotherm = None,
    pressure: list = None,
):
    loading = []
    for p in pressure:
        loading.append(
            original_isotherm.loading_at(p)
        )

    return pg.PointIsotherm(
        pressure=pressure,
        loading=loading,

        material=str(original_isotherm.material),
        adsorbate=str(original_isotherm.adsorbate),
        temperature=str(original_isotherm.temperature),

        pressure_mode=original_isotherm.pressure_mode,
        pressure_unit=original_isotherm.pressure_unit,
        loading_basis=original_isotherm.loading_basis,
        loading_unit=original_isotherm.loading_unit,
        material_basis=original_isotherm.material_basis,
        material_unit=original_isotherm.material_unit,
        temperature_unit=original_isotherm.temperature_unit,
    )


def isotherm_splitter(
    isotherm,
    linear_pressure_lists,
):
    isotherm_splits = {}
    n=0
    for pressure_list in linear_pressure_lists:
        isotherm_splits[n] = isotherm_from_pressures(
            isotherm,
            pressure_list,
        )
        n+=1

    return isotherm_splits


def DRtransform(
    isotherm,
    exp: int = 2,
):
    loading = list(isotherm.loading())
    pressure = list(isotherm.pressure())
    log_v = np.log10(loading)
    log_p_exp = (-np.log10(pressure))**exp
    log_p_exp = np.flip(log_p_exp)
    return {
        'log_p_exp': log_p_exp,
        'log_v': log_v
    }


def drop_curved(
    isotherm_splits,
    exp: int = 2,
    tolerance: float = 0.1,
):
    filtered_splits = {}
    for i in isotherm_splits:
        isotherm = isotherm_splits[i]
        dr = DRtransform(isotherm, exp)
        not_curved = curvature_filter(
            dr['log_p_exp'], dr['log_v'],
            tolerance,
        )
        if not_curved:
            filtered_splits[i] = isotherm


    return filtered_splits


def DRresults(
    isotherm_splits,
    isotherm,
):
    results_dict = {}
    n=0
    for s in isotherm_splits:
        split = isotherm_splits[s]
        pressure = list(split.pressure())
        try:
            results = pgc.dr_da_plots.dr_plot(
                isotherm,
                p_limits=[min(pressure), max(pressure)],
            )
        except ParameterError:
            continue
        except CalculationError:
            continue

        results['num_points'] = len(pressure)
        results['pressure'] = pressure
        results_dict[n] = results
        n+=1

    return results_dict


def DRresults_filter(
    results_dict,
    isotherm,
    min_corr_coef: float = 0.99,
):
    results_filtered = {}
    for r in results_dict:
        result = results_dict[r]

        good_correlation = correlation_filter(
            result['corr_coef'],
            min_corr_coef)
        good_pore_volume = volume_filter(
            result['limiting_micropore_capacity'],
            max(isotherm.loading())
        )

        if (good_correlation and good_pore_volume):
            results_filtered[r] = result

    return results_filtered


def DRoptimum(
    results_dict
):
    if len(results_dict) == 0:
        raise ValueError(
            'No data to use! Closing program'
        )
        return

    num_points = 0
    for r in results_dict:
        if results_dict[r]['num_points'] > num_points:
            num_points = results_dict[r]['num_points']
            result = results_dict[r]
    return result


def analyseDR(
    isotherm,
    num_points: int = 10,
    p_limits: tuple[float, float] = [0, 0.1],
    corr_coef: float = 0.99,
    curvature: float = 0.1,
    verbose: bool = False,
    output_dir: str = './DR/'
):
    pressure_lists = split_pressures(
        isotherm,
        p_limits=p_limits,
    )
    isotherm_splits = isotherm_splitter(
        isotherm,
        pressure_lists,
    )
    isotherm_splits = drop_curved(
        isotherm_splits,
        tolerance=curvature,
    )
    results = DRresults(
        isotherm_splits,
        isotherm
    )
    results_filtered = DRresults_filter(
        results,
        isotherm
    )
    print(len(results_filtered))
    result = DRoptimum(results_filtered)

    if verbose:
        print(
            f'\nOptimum DR result selected from '
            f'{len(results_filtered)} filtered results.'
            f' Optimum result:\n'
        )
        pgc.dr_da_plots.dr_plot(
            isotherm,
            p_limits=[
                min(result['pressure']),
                max(result['pressure'])
            ],
            verbose=True
        )
        plt.show()

    if output_dir is not None:
        import os
        import pandas as pd
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pd.DataFrame(results_filtered).to_csv(
            f'{output_dir}filtered.csv',
            index=False
        )

    return result
