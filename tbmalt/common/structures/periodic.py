"""Deal with periodic conditions."""
import torch
import numpy as np
from typing import Union, List
from tbmalt.common.batch import pack
Tensor = torch.Tensor
_bohr = 0.529177249


class Periodic:
    """Calculate the translation vectors for cells for 3D periodic boundary condition.

    Arguments:
        latvec: Lattice vector describing the geometry of periodic system,
            with Bohr as unit.
        cutoff: Interaction cutoff distance for reading  SK table.

    Keyword Args:
        distance_extention: Extention of cutoff in SK tables to smooth the tail.
        positive_extention: Extension for the positive lattice vectors.
        negative_extention: Extension for the negative lattice vectors.

    Return:
        cutoff: Global cutoff for the diatomic interactions, unit is Bohr.
        cellvol: Volume of the unit cell.
        reccellvol: Volume of the reciprocal lattice unit cell.
        cellvec: Cell translation vectors in relative coordinates.
        rcellvec: Cell translation vectors in absolute units.
        ncell: Number of lattice cells.

    Examples:
        >>> from periodic import Periodic
        >>> import torch
    """

    def __init__(self, system: object, latvec: Tensor,
                 cutoff: Union[Tensor, float], **kwargs):
        self.system = system
        self.latvec, self.cutoff = self._check(latvec, cutoff, **kwargs)
        self._positions_check(**kwargs)
        dist_ext = kwargs.get('distance_extention', 1.0)

        # Global cutoff for the diatomic interactions
        self.cutoff = self.cutoff + dist_ext

        self.invlatvec = self._inverse_lattice()

        self.cellvol, self.cellvec, self.rcellvec = self.get_cell_translations(**kwargs)

        self.positions_vec, self.periodic_distances = self._get_periodic_distance()

    def _check(self, latvec, cutoff, **kwargs):
        """Check dimension, type of lattice vector and cutoff."""
        _mask = self.system.is_periodic
        # default lattice vector is from system, therefore default unit is bohr
        unit = kwargs.get('unit', 'bohr')

        # molecule will be padding with zeros, here select latvec for solid
        if type(latvec) is list:
            latvec = pack(latvec)
        elif type(latvec) is not Tensor:
            raise TypeError('Lattice vector is tensor or list of tensor.')

        if latvec.dim() == 2:
            latvec = latvec.unsqueeze(0)[_mask]
        elif latvec.dim() == 3:
            latvec = latvec[_mask]
        else:
            raise ValueError('lattice vector dimension should be 2 or 3')

        # currently, cutoff is the same for all systems
        if type(cutoff) is Tensor:
            if cutoff.dim() == 0:
                cutoff = cutoff.unsqueeze(0)
            elif cutoff.dim() >= 2:
                raise ValueError(
                    'cutoff should be 0, 1 dimension tensor or float')
        elif type(cutoff) is not float:
            raise TypeError('cutoff should be tensor or float')

        if unit in ('angstrom', 'Angstrom'):
            latvec = latvec / _bohr
            cutoff = cutoff / _bohr
        elif unit not in ('bohr', 'Bohr'):
            raise ValueError('unit is either angstrom or bohr')

        return latvec, cutoff

    def _positions_check(self, **kwargs):
        """Check positions type (fraction or not) and unit."""
        unit = kwargs.get('unit', 'angstrom')

        # transfer from fraction to Bohr unit positions
        position_pe = self.system.positions[self.system.is_periodic]
        is_frac = pack([abs(ipos).lt(1.).all() for ipos in position_pe])
        position_pe[is_frac] = torch.matmul(
            position_pe[is_frac], self.latvec[is_frac])

        # transfer other periodic positions (non-fraction) to bohr
        if unit in ('angstrom', 'Angstrom'):
            position_pe[~is_frac] = position_pe[~is_frac] / _bohr
        elif unit not in ('bohr', 'Bohr'):
            raise ValueError('Please select either angstrom or bohr')

        self.system.positions[self.system.is_periodic] = position_pe

    def get_cell_translations(self, **kwargs):
        """Get."""
        pos_ext = kwargs.get('positive_extention', 1)
        neg_ext = kwargs.get('negative_extention', 1)

        # Unit cell volume
        cellvol = abs(torch.det(self.latvec))
        _tmp = torch.floor(self.cutoff * torch.norm(self.invlatvec, dim=-1))
        ranges = torch.stack([-(neg_ext + _tmp), pos_ext + _tmp])

        # Length of the first, second and third column in ranges
        leng = ranges[1, :].long() - ranges[0, :].long() + 1

        # Cell translation vectors in relative coordinates
        cellvec = pack([torch.stack([
            torch.linspace(iran[0, 0], iran[1, 0],
                           ile[0]).repeat_interleave(ile[2] * ile[1]),
            torch.linspace(iran[0, 1], iran[1, 1],
                           ile[1]).repeat(ile[0]).repeat_interleave(ile[2]),
            torch.linspace(iran[0, 2], iran[1, 2],
                           ile[2]).repeat(ile[0] * ile[1])])
            for ile, iran in zip(leng, ranges.transpose(1, 0))])

        rcellvec = pack([(ilv.transpose(0, 1) @ icv.T.unsqueeze(-1)).squeeze(-1)
                         for ilv, icv in zip(self.latvec, cellvec)])

        return cellvol, cellvec, rcellvec

    def _get_periodic_distance(self):
        """Get distances between central cell and neighbour cells."""
        positions = self.rcellvec.unsqueeze(2) + self.system.positions.unsqueeze(1)
        size_system = self.system.size_system

        # positions_vec = (positions.transpose(1, 0).unsqueeze(-3) -
        #                  positions.transpose(1, 0).unsqueeze(-2)).transpose(1, 0)
        positions_vec = (positions.unsqueeze(-3) - self.system.positions.unsqueeze(1).unsqueeze(-2))

        return positions_vec, pack([torch.sqrt(((
            ipos[:, :inat].repeat(1, inat, 1) - torch.repeat_interleave(
                icp[:inat], inat, 0)) ** 2).sum(-1)).reshape(-1, inat, inat)
            for ipos, icp, inat in zip(positions, self.system.positions, size_system)])

    def _inverse_lattice(self):
        """Get inverse lattice vectors."""
        return torch.transpose(torch.solve(torch.eye(
                self.latvec.shape[-1]), self.latvec)[0], -1, -2)

    def get_reciprocal_volume(self):
        """Get reciprocal lattice unit cell volume."""
        return abs(torch.det(2 * np.pi * (self.invlatvec.transpose(0, 1))))

    @property
    def neighbour(self):
        """Get neighbour list according to periodic boundary condition."""
        return self.periodic_distances.le(self.cutoff)
