"""Parameters."""
from typing import Tuple, Union, Optional


class Parameter:
    """DFTB parameters."""

    def __init__(self, dftb_params=None, constant_params=None, ml_params=None):
        self.dftb_params = self.dftb_params(dftb_params)
        self.constant_params = self.constant_params(constant_params)
        if ml_params is not None:
            self.ml_params = self.ml_params(ml_params)

    def ml_params(self, ml_params: Union[bool, dict]) -> dict:
        """Return machine learning parameters."""
        _ml_params = {
            'lr': 0.1,
            'steps': 3,  # -> training steps
            'loss_function': 'MSELoss',  # MSELoss, L1Loss
            'optimizer': 'Adam',  # SCG, Adam
            'compression_radii_min': 1.5,
            'compression_radii_max': 9.0
            }

        if type(ml_params) is dict:
            _ml_params.update(ml_params)

        return _ml_params

    def constant_params(self, constant_params) -> dict:
        """Return constant parameter."""
        _constant_params = {'bohr': 0.529177249}

        if constant_params is not None:
            if type(constant_params) is dict:
                _constant_params.update(constant_params)
            else:
                raise TypeError('constant_params should be None or dict')

        return _constant_params

    def dftb_params(self, dftb_params) -> dict:
        """Parameter for DFTB."""
        _dftb_params = {
            'mix': 'Anderson',  # -> Anderson, Simple
            'scc': 'scc',  # 'scc', 'nonscc', 'xlbomd'
            'maxiter': 60  # -> max SCC loop
            }

        if dftb_params is not None:
            print('dftb_params', type(dftb_params))
            if type(dftb_params) is dict:
                _dftb_params.update(dftb_params)
            else:
                raise TypeError('dftb_params should be None or dict')

        return _dftb_params
