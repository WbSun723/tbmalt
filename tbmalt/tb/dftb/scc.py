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

        self.properties = Properties(properties, self.system, self.qzero,
                                     self.charge, self.over, self.rho)

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
            # self.rho = torch.matmul(c_scaled, c_scaled.transpose(1, 2))

            # calculate mulliken charges for each system in batch
            q_new = mulliken(self.over[self.mask, :this_size, :this_size],
                             self.rho, self.atom_orbitals[self.mask])

            # last mixed charge is the current step now
            if self.scc == 'nonscc':
                self.charge = q_new
            else:
                self.qmix, self.converge = self.mixer(q_new)
                self._update_charge(self.qmix)

                if (self.converge == True).all():
                    break  # -> all system reach convergence
        return self.charge

    def _update_charge(self, qmix):
        """Update charge according to convergence in last step."""
        self.charge[self.mask] = qmix
        self.mask = ~self.converge


    # def scf_npe_scc(self):
    #     """SCF for non-periodic-ML system with scc.

    #     atomind is the number of atom, for C, lmax is 2, therefore
    #     we need 2**2 orbitals (s, px, py, pz), then define atomind2
    #     """
    #     from tbmalt.common.tmp import EigenSolver
    #     from tbmalt.common.tmp2 import Anderson
    #     eigen = EigenSolver()
    #     mixer = Anderson()
    #     gmat = Gamma(self.skt.U, self.system.distances).gamma

    #     # qatom = self.analysis.get_qatom(self.atomname, ibatch)

    #     # qatom here is 2D, add up the along the rows
    #     nelectron = self.qzero.sum(axis=1)
    #     qzero = self.qzero.clone()
    #     q_mixed = qzero.clone()  # q_mixed will maintain the shape unchanged
    #     self.maxiter = 1
    #     for iiter in range(self.maxiter):
    #         shift_ = torch.stack(
    #             [(im - iz) @ ig for im, iz, ig in zip(
    #                 self.charge, self.qzero, gmat)])

    #         # repeat shift according to number of orbitals
    #         shiftorb_ = pack([ishif.repeat_interleave(iorb) for iorb, ishif in
    #                           zip(self.atom_orbitals, shift_)])
    #         shift_mat = torch.stack([torch.unsqueeze(ishift, 1) + ishift
    #                                  for ishift in shiftorb_])

    #         # To get the Fock matrix
    #         dim_ = shift_mat.shape[-1]   # the new dimension of max orbitals
    #         fock = self.ham + 0.5 * self.over * shift_mat

    #         # Calculate the eigen-values & vectors via a Cholesky decomposition
    #         epsilon, C = eigen.eigen(fock, self.over, True)

    #         # Calculate the occupation of electrons via the fermi method
    #         occ, nocc = fermi(epsilon, nelectron)

    #         # build density according to occ and eigenvector
    #         C_scaled = torch.sqrt(occ).unsqueeze(1).expand_as(C) * C

    #         # batch calculation of density, normal code: C_scaled @ C_scaled.T
    #         rho = torch.matmul(C_scaled, C_scaled.transpose(1, 2))

    #         # calculate mulliken charges for each system in batch
    #         q_new = mulliken(self.over, rho, self.atom_orbitals)

    #         # Last mixed charge is the current step now
    #         q_mixed = mixer(q_new, q_mixed)

    #         if iiter > 20:
    #             break

    #     # return eigenvalue and charge
    #     self.charge = q_mixed
    #     self.rho = rho
    #     # shiftorb = pack([
    #     #     ishif.repeat_interleave(iorb) for iorb, ishif in zip(
    #     #         self.atind, self.para['shift'])])
    #     # shift_mat = torch.stack([torch.unsqueeze(ish, 1) + ish for ish in shiftorb])
    #     # fock = self.ham + 0.5 * self.over * shift_mat
    #     # self.para['eigenvalue'], self.para['eigenvec'] = \
    #     #     self.eigen.eigen(fock, self.over, self.batch, self.atind)

    #     # # return occupied states
    #     # self.para['occ'], self.para['nocc'] = \
    #     #     self.elect.fermi(self.para['eigenvalue'], nelectron, self.para['tElec'])

    #     # # return density matrix
    #     # C_scaled = torch.sqrt(self.para['occ']).unsqueeze(1).expand_as(
    #     #     self.para['eigenvec']) * self.para['eigenvec']
    #     # self.para['denmat'] = torch.matmul(C_scaled, C_scaled.transpose(1, 2))
    #     return C
