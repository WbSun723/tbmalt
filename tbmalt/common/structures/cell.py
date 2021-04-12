"""
Created on Wed Feb 10 10:54:18 2021

@author: gz_fan
"""
import torch
from typing import Union, List
from tbmalt.common.batch import pack
Tensor = torch.Tensor
_bohr = 0.529177249


class Cell:
    """Cell class."""

    def __init__(self, cell: Union[Tensor, List[Tensor]], pbc=None, frac=None, **kwargs):
        self.cell, self.pbc, self.is_periodic, self.is_frac = self._check_cell(cell, pbc, frac)

    def _check_cell(self, cell, pbc, frac, **kwargs):
        """Check cell type and dimension, transfer to batch tensor."""
        unit = kwargs.get('unit', 'angstrom')
        if type(cell) is list:
            cell = pack(cell)
        elif type(cell) is Tensor:
            if cell.dim() == 2:
                cell = cell.unsqueeze(0)
            elif cell.dim() < 2 or cell.dim() > 3:
                raise ValueError('input cell dimension is not 2 or 3')

        # non periodic in cell will be zero
        is_periodic = torch.stack([ic.ne(0).any() for ic in cell])

        # some systems in batch is periodic
        if is_periodic.any():
            if pbc is None:
                pbc = torch.ones(*cell.shape, dtype=bool)
            elif type(pbc) is list:
                assert type(pbc[0]) is Tensor
                assert pbc[0].dtype is torch.bool
                pbc = pack(pbc)
            elif type(pbc) is Tensor:
                assert pbc.dtype is torch.bool
                if pbc.dim() == 1:
                    pbc = pbc.unsqueeze(0)
                elif pbc.dim() > 3:
                    raise ValueError('pbc dimension error')
            else:
                raise TypeError('pbc is torch.Tensor or list of torch.Tensor.')

        # some systems in batch is fraction coordinate
        if frac is not None:
            is_frac = torch.stack([ii.ne(0).any() for ii in frac]) & is_periodic
        else:
            is_frac = torch.zeros(cell.size(0), dtype=bool)

        # transfer positions from angstrom to bohr
        if unit in ('angstrom', 'Angstrom'):
            cell = cell / _bohr
        elif unit not in ('bohr', 'Bohr'):
            raise ValueError('Please select either angstrom or bohr')

        return cell, pbc, is_periodic, is_frac

    @property
    def get_reciprocal_cell(self):
        """Get reciprocal cell."""
