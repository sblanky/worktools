import pygaps as pg
import pygaps.parsing as pgp
import pygaps.characterisation as pgc
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import optimize, stats
import numpy as np
import pprint
from collections import OrderedDict
import pandas as pd
import os
from pathlib import Path

from pygaps.utilities.exceptions import CalculationError
from pygaps.utilities.exceptions import ParameterError



def log_p_exp(
    pressure,
    exp
):
    return (-np.log10(pressure))**exp


class DubininResult:
    """
    Store all possible Dubinin results from an isotherm,
    as well as curvature, and other information for filtering
    later.
    """
    def __init__(
        self,
        isotherm,
        exp: float = None,
        **kwargs,
    ):
        self.adsorbate = isotherm.adsorbate
        self.iso_temp = isotherm.temperature
        self.molar_mass = self.adsorbate.molar_mass()
        self.liquid_density = self.adsorbate.liquid_density(self.iso_temp)
        gas_density = 1e-3 * (self.molar_mass/22.4)
        self.density_conversion_factor = gas_density / self.liquid_density

        self.pressure = isotherm.pressure(
            branch='ads',
            pressure_mode='relative',
        )
        self.loading = isotherm.loading_at(
            pressure=list(self.pressure),
            branch='ads',
            pressure_mode='relative',
            loading_unit='cm3',
            loading_basis='volume_gas',
            material_unit='g',
            material_basis='mass',
        )
        self.total_pore_capacity = max(isotherm.loading())


        self.log_v = np.log10(self.loading)

        bounds = kwargs.get('bounds', [1,3])

        def dr_fit(exp, ret=False):
            slope, intercept, corr_coef, p_val, stderr = stats.linregress(
                log_p_exp(self.pressure, exp), self.log_v
            )
            if ret:
                return slope, intercept, corr_coef
            return stderr

        if exp is None:
            res = optimize.minimize_scalar(
                dr_fit,
                bounds=bounds,
                method='bounded'
            )
            if not res.success:
                raise CalculationError(
                    'Could not obtain a linear fit on the data.'
                )
            self.exp = res.x
        else:
            self.exp = exp

        self.log_p_exp = log_p_exp(self.pressure, self.exp)

        self._compute_dubinin_data(
            self.pressure,
            self.loading,
        )


    def _compute_dubinin_data(
        self,
        pressure,
        loading,
    ):

        try:
            spline = PchipInterpolator(
                np.flip(self.log_p_exp),
                np.flip(self.log_v)
            )
            second_derivative = spline.derivative(nu=2)
        except ValueError as e:
            print(e)
            pass

        num_points = len(pressure)
        self.result = {}
        for i in range(num_points):
            for j in range(i+3, num_points):
                self.result[i, j] = {}
                pressure_range=[self.pressure[i], self.pressure[j]]

                try:
                    (
                        microp_capacity,
                        potential,
                        _,
                        slope,
                        intercept,
                        _, _,
                        corr_coef,
                    ) = pgc.dr_da_plots.da_plot_raw(
                        self.pressure,
                        self.loading,
                        self.iso_temp,
                        self.molar_mass,
                        self.liquid_density,
                        exp = self.exp,
                        p_limits = pressure_range,
                    )
                    self.result[i, j] = {
                        'microp_capacity': microp_capacity,
                        'microp_volume': microp_capacity * self.density_conversion_factor,
                        'potential': potential,
                        'slope': slope,
                        'intercept': intercept,
                        'corr_coef': corr_coef
                    }


                except CalculationError:
                    continue

                log_p_exp = self.log_p_exp[i:j]
                try:
                   derivs = [
                       min(second_derivative(log_p_exp)),
                       max(second_derivative(log_p_exp))
                   ]
                   deriv_change = abs(derivs[1]-derivs[0])
                   x_range = abs(log_p_exp[0]-log_p_exp[-1])
                   relative_change = deriv_change / np.log10(x_range)
                   self.result[i, j]['curvature'] = abs(relative_change)
                except ValueError:
                    continue

                self.result[i, j]['point_count'] = j - i

                self.result[i, j]['pressure_range'] = pressure_range



class DubininFilteredResults:

    def __init__(
        self,
        dubinin_result,
        **kwargs,
    ):
        self.__dict__.update(dubinin_result.__dict__)
        self.filter_params = kwargs

        p_limits = kwargs.get('p_limits', [0, 0.1])
        self._filter(
            'pressure_range',
            lambda x: x[0] < p_limits[0] or x[1] > p_limits[1]
        )

        curvature_limit = kwargs.get('curvature_limit', 1)
        self._filter(
            'curvature',
            lambda x: abs(x) > curvature_limit
        )
        print(curvature_limit)

        max_capacity = kwargs.get('max_capacity', self.total_pore_capacity)
        self._filter(
            'microp_capacity',
            lambda x: x > max_capacity

        )

        min_points = kwargs.get('min_points', 10)
        self._filter(
            'point_count',
            lambda x: x < min_points
        )

        min_corr_coef = kwargs.get('min_corr_coef', 0.999)
        self._filter(
            'corr_coef',
            lambda x: abs(x) < min_corr_coef
        )

        self._stats()

        optimum = 1e10
        for r in self.result:
            if self.result[r]['microp_volume'] < optimum:
                optimum = self.result[r]['microp_volume']
                self.optimum = self.result[r]

    def _filter(
        self,
        key,
        criteria,
    ):
        to_remove = []
        for r in self.result:
            result = self.result[r]
            if criteria(result[key]):
                to_remove.append(r)

        for r in to_remove:
            del self.result[r]

    def _sort(self):
        self.result = OrderedDict(sorted(
            self.result.items(),
            key=lambda x: (x[1]['point_count'], x[1]['corr_coef'])
        )
        )

    def _stats(self):
        volumes = [ x['microp_volume'] for x in self.result.values() ]
        optimised_volume = min(volumes)
        self.mean = np.mean(volumes)
        self.stddev = np.std(volumes)

    def export(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        pd.DataFrame(self.result).transpose().to_csv(
            f'{filepath}filtered_results.csv',
            index=False
        )


def analyseDR(
    isotherm,
    output_dir=None,
    verbose=False,
):
    if output_dir is None:
        output_dir = f'./dubinin/'

    output_subdir = f'{output_dir}{isotherm.material}/'

    result = DubininResult(
        isotherm,
        #**{'bounds':[1,3.5]}
    )
    kwargs = {}
    kwargs['curvature_limit'] = 1
    filtered_result = DubininFilteredResults(
        result,
        **kwargs
    )
    filtered_result.export(output_subdir)
    if verbose:
        plt.scatter(
            filtered_result.log_p_exp,
            filtered_result.log_v,
            fc=None, ec='k',
        )
        x = np.linspace(
            filtered_result.log_p_exp[0],
            filtered_result.log_p_exp[-1])
        y = filtered_result.optimum['slope'] * x + filtered_result.optimum['intercept']
        plt.plot(x, y)
        plt.show()


if __name__ == "__main__":
    Path = '/home/pcxtsbl/CodeProjects/labcore_upload/robert/aif/'
    file = 'ACC2600.aif'
    isotherm = pgp.isotherm_from_aif(f'{Path}{file}')
    analyseDR(isotherm)
