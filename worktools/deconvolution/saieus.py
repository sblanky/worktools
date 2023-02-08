import pandas as pd

class saieus_results:

    _required_params = [
        'material',
        'model',
        'lambda',
        'stdev',
        'w', 'dV', 'Vcum', 'dS', 'Scum',
        'PP0', 'Q', 'Fit'

    ]


    def __init__(
        self,
    ):
        if None in self._required_params:
            raise ParameterError(
                'SAIEUS analysis must have following parameters '
                f'{self._required_params}.'
            )


    @property
    def to_dict(self) -> dict:

