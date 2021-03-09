"""DFTB calculator.

implement pytorch to DFTB
"""
import torch
import tbmalt.common.maths as maths
from tbmalt.tb.electrons import fermi
from tbmalt.tb.properties import mulliken
from tbmalt.common.maths.mixer import Simple, Anderson
from tbmalt.tb.properties import Properties
from tbmalt.common.batch import pack
from tbmalt.tb.electrons import Gamma
from tbmalt.tb.coulomb import Coulomb


class Scc:
    """Self-consistent charge density functional tight binding method.

    Arguments:
        system: Object contains geometry and orbital information.
        skt: Object contains SK data.
        parameter: Object which return DFTB and ML parameters.

    Examples:
        >>> from ase.build import molecule as molecule_database
        >>> from tbmalt.common.structures.system import System
        >>> from tbmalt.io.loadskf import IntegralGenerator
        >>> from tbmalt.tb.sk import SKT
        >>> from tbmalt.tb.dftb.scc import SCC
        >>> from tbmalt.common.parameter import DFTBParams
        >>> molecule = System.from_ase_atoms([molecule_database('CH4')])
        >>> sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
        >>> skt = SKT(molecule, sktable)
        >>> parameter = DFTBParams()
        >>> scc = SCC(molecule, skt, parameter)
        >>> scc.charge
        >>> tensor([[4.3054, 0.9237, 0.9237, 0.9237, 0.9237]])

    """

    def __init__(self, system: object, skt: object, parameter: object,
                 coulomb=None, periodic=None, properties=[], **kwargs):
        self.system = system
        self.skt = skt
        self.parameter = parameter
        self.coulomb = coulomb
        self.periodic = periodic

        self.ham, self.over = skt.H, skt.S
        self._init_scc(**kwargs)

        self._scc()

        self.properties = Properties(properties, self.system, self.qzero,
                                     self.charge, self.over, self.rho)

    def _init_scc(self, **kwargs):
        """Initialize parameters for (non-) SCC DFTB calculations."""
        # basic DFTB parameters
        self.scc = self.parameter.scc
        self.maxiter = self.parameter.maxiter if self.scc == 'scc' else 1
        self.mask = torch.tensor([True]).repeat(self.ham.shape[0])
        self.atom_orbitals = self.system.atom_orbitals

        # intial charges
        self.qzero = self.system.get_valence_electrons(self.ham.dtype)
        self.charge = kwargs.get('charge') if self.scc == 'xlbomd' else self.qzero.clone()
        self.nelectron = self.qzero.sum(axis=1)

        # get the mixer
        self.mix = self.parameter.mix
        if self.mix in ('Anderson', 'anderson'):
            self.mixer = Anderson(self.charge, return_convergence=True)
        elif self.mix in ('Simple', 'simple'):
            self.mixer = Simple(self.charge, return_convergence=True)

        if self.system.periodic:
            assert self.coulomb is not None
            assert self.periodic is not None
            self.distances = self.periodic.periodic_distances
            self.u = self._expand_u(self.skt.U)
        else:
            self.distances = self.system.distances
            self.u = self.skt.U

        if self.scc in ('scc', 'xlbomd'):
            self.gamma = Gamma(self.u, self.distances).gamma
        else:
            self.gamma = torch.zeros(*self.qzero.shape)

        self.inv_dist = self._inv_distance(self.system.distances)

        self.shift = self._get_shift()

    def _scc(self, ibatch=[0]):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        for iiter in range(self.maxiter):
            # get shift and repeat shift according to number of orbitals
            shift_ = self._update_shift()
            shiftorb_ = pack([ishif.repeat_interleave(iorb) for iorb, ishif in
                              zip(self.atom_orbitals[self.mask], shift_)])
            shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
                                     for ishift in shiftorb_])

            # H0 + 0.5 * S * G
            this_size = shift_mat.shape[-1]   # the new shape
            fock = self.ham[self.mask, :this_size, :this_size] + \
                0.5 * self.over[self.mask, :this_size, :this_size] * shift_mat

            # calculate the eigen-values & vectors via a Cholesky decomposition
            epsilon, C = maths.eighb(fock, self.over[self.mask, :this_size, :this_size])

            # calculate the occupation of electrons via the fermi method
            occ, nocc = fermi(epsilon, self.nelectron[self.mask])

            # eigenvector with Fermi-Dirac distribution
            C_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(C) * C
            self.rho = C_scaled @ C_scaled.transpose(1, 2)  # -> density

            # calculate mulliken charges for each system in batch
            q_new = mulliken(self.over[self.mask, :this_size, :this_size],
                             self.rho, self.atom_orbitals[self.mask])

            # last mixed charge is the current step now
            self.qmix, self.converge = self.mixer(q_new)
            self._update_charge(self.qmix)

            if (self.converge == True).all():
                break  # -> all system reach convergence

    def _update_charge(self, qmix):
        """Update charge according to convergence in last step."""
        self.charge[self.mask] = qmix
        self.mask = ~self.converge

    def _inv_distance(self, distance):
        _inv_distance = torch.zeros(*distance.shape)
        mask = distance.ne(0.0)
        _inv_distance[mask] = 1.0 / distance[mask]
        return _inv_distance

    def _get_shift(self):
        """Return shift term for periodic and non-periodic."""
        if not self.system.periodic:
            return self.inv_dist - self.gamma
        else:
            return self.coulomb.invrmat - self.gamma

    def _update_shift(self):
        """Update shift."""
        return torch.stack([(im - iz) @ ig for im, iz, ig in zip(
            self.charge[self.mask], self.qzero[self.mask], self.shift[self.mask])])

    def _expand_u(self, u):
        """Expand Hubbert U for periodic system."""
        shape_cell = self.distances.shape[1]
        return u.repeat(1, shape_cell, 1)
