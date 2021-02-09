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
                 properties=[], **kwargs):
        self.system = system
        self.skt = skt
        self.parameter = parameter

        self.ham, self.over = skt.H, skt.S
        self._init_scc(**kwargs)

        self._scc_npe()

        self.properties = Properties(properties, self.system, self.qzero,
                                     self.charge, self.over, self.rho)

    def _init_scc(self, **kwargs):
        """Initialize parameters for (non-) SCC DFTB calculations."""
        # basic DFTB parameters
        self.scc = self.parameter.scc
        self.maxiter = self.parameter.maxiter if self.scc == 'scc' else 1
        self.mask = torch.tensor([True]).repeat(self.ham.shape[0])
        self.atom_orbitals = self.system.atom_orbitals
        self.size_system = self.system.size_system  # -> atoms in each system

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

    def _scc_npe(self, ibatch=[0]):
        """SCF for non-periodic-ML system with scc.

        atomind is the number of atom, for C, lmax is 2, therefore
        we need 2**2 orbitals (s, px, py, pz), then define atomind2
        """
        if self.scc in ('scc', 'xlbomd'):
            gamma = Gamma(self.skt.U, self.system.distances).gamma
        else:
            gamma = torch.zeros(*self.qzero.shape)

        for iiter in range(self.maxiter):
            shift_ = torch.stack(
                [(im - iz) @ ig for im, iz, ig in zip(
                    self.charge[self.mask], self.qzero[self.mask], gamma[self.mask])])

            # repeat shift according to number of orbitals
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
            if self.scc == 'xlbomd':
                print(self.scc, self.charge)

            if (self.converge == True).all():
                break  # -> all system reach convergence

    def _update_charge(self, qmix):
        """Update charge according to convergence in last step."""
        self.charge[self.mask] = qmix
        self.mask = ~self.converge
