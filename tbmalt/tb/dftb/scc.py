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
        self.params = parameter

        self.ham, self.over = skt.H, skt.S
        self._init_scc(**kwargs)

        self._scc_npe()

        # if 'dipole' in properties:
        self.dipole = self._dipole()
        # if 'homo_lumo' or 'gap' in properties:
        self.homo_lumo = self._homo_lumo()
        self.cpa = self._cpa()
        # self.properties = Properties(properties, self.system, self.qzero,
        #                              self.charge, self.over, self.rho)

    def _init_scc(self, **kwargs):
        """Initialize parameters for (non-) SCC DFTB calculations."""
        # basic DFTB parameters
        self.scc = self.params.dftb_params['scc']
        self.maxiter = self.params.dftb_params['maxiter'] if self.scc == 'scc' else 1
        self.mask = torch.tensor([True]).repeat(self.ham.shape[0])
        self.atom_orbitals = self.system.atom_orbitals
        self.size_system = self.system.size_system  # -> atoms in each system

        # intial charges
        self.qzero = self.system.get_valence_electrons(self.ham.dtype)
        self.charge = kwargs.get('charge') if self.scc == 'xlbomd' else self.qzero.clone()
        self.nelectron = self.qzero.sum(axis=1)

        # get the mixer
        self.mix = self.params.dftb_params['mix']
        if self.mix in ('Anderson', 'anderson'):
            self.mixer = Anderson(self.charge, return_convergence=True)
        elif self.mix in ('Simple', 'simple'):
            self.mixer = Simple(self.charge, return_convergence=True)

    def _scc_npe(self):
        """SCF for non-periodic-ML system with scc."""
        if self.scc in ('scc', 'xlbomd'):
            gamma = Gamma(self.skt.U, self.system.distances).gamma
        else:
            gamma = torch.zeros(*self.system.distances.shape)

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
            epsilon, eigvec = maths.eighb(fock, self.over[self.mask, :this_size, :this_size])

            # calculate the occupation of electrons via the fermi method
            occ, nocc = fermi(epsilon, self.nelectron[self.mask])

            # eigenvector with Fermi-Dirac distribution
            c_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(eigvec) * eigvec
            self.rho = c_scaled @ c_scaled.transpose(1, 2)  # -> density

            # calculate mulliken charges for each system in batch
            q_new = mulliken(self.over[self.mask, :this_size, :this_size],
                             self.rho, self.atom_orbitals[self.mask])

            # last mixed charge is the current step now
            if self.scc == 'nonscc':
                self.charge = q_new
            else:
                if iiter == 0:
                    self.eigenvalue = torch.zeros(*epsilon.shape)
                    self.nocc = torch.zeros(*nocc.shape)
                    self.density = torch.zeros(*self.rho.shape)
                self.qmix, self.converge = self.mixer(q_new)
                self._update_scc(self.qmix, epsilon, nocc)

                if (self.converge == True).all():
                    break  # -> all system reach convergence

        return self.charge

    def _update_scc(self, qmix, epsilon, nocc):
        """Update charge according to convergence in last step."""
        self.charge[self.mask] = qmix
        self.eigenvalue[self.mask, :epsilon.shape[1]] = epsilon
        self.nocc[self.mask] = nocc
        self.density[self.mask, :self.rho.shape[1], :self.rho.shape[2]] = self.rho
        self.mask = ~self.converge

    def _dipole(self):
        """Return dipole moments."""
        return torch.sum((self.qzero - self.charge).unsqueeze(-1) *
                         self.system.positions, 1)

    def _homo_lumo(self):
        """Return dipole moments."""
        # get HOMO-LUMO, not orbital resolved
        return torch.stack([
            ieig[int(iocc) - 1:int(iocc) + 1] * 27.2113845
            for ieig, iocc in zip(self.eigenvalue, self.nocc)])

    def _onsite_population(self):
        """Get onsite population for CPA DFTB.

        sum density matrix diagnal value for each atom
        """
        ao = self.system.atom_orbitals
        nb = self.system.size_batch
        ns = self.system.size_system
        acum = torch.cat([torch.zeros(ao.shape[0]).unsqueeze(0),
                          torch.cumsum(ao, dim=1).T]).T.long()
        denmat = [idensity.diag() for idensity in self.density]
        # get onsite population
        return pack([torch.stack([torch.sum(denmat[ib][acum[ib][iat]: acum[ib][iat + 1]])
                    for iat in range(ns[ib])]) for ib in range(nb)])

    def _cpa(self):
        """Get onsite population for CPA DFTB.

        J. Chem. Phys. 144, 151101 (2016)
        """
        onsite = self._onsite_population()
        nat = self.system.size_system
        numbers = self.system.numbers

        return pack([1.0 + (onsite[ib] - self.qzero[ib])[:nat[ib]] / numbers[ib][:nat[ib]]
                     for ib in range(self.system.size_batch)])
