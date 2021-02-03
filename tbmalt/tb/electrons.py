"""Electronic calculations."""
import torch
import torch.nn.functional as F
from tbmalt.common.batch import pack
Tensor = torch.Tensor


class Gamma:
    """Build the gamma in second-order term of DFTB.

    Arguments:
        U: non orbital resolved Hubbert
        distance: distance of single system or batch, unit is bohr

    Keyword Args:
        gamma_type: slater or gaussian type, the option with short, e.g.
            slater_short will not include 1 / distance term.
    """

    def __init__(self, U: Tensor, distance: Tensor, **kwargs) -> Tensor:
        self.U = U
        self.distance = distance
        self.gamma_type = kwargs.get('gamma_type', 'slater')

        # call gamma funcitons
        if self.gamma_type in ('slater', 'slater_short'):
            self.gamma = self.gamma_slater()
        elif self.gamma_type == 'gaussian':
            self.gamma = self.gamma_gaussian()

    def gamma_slater(self):
        """Build the Slater type gamma in second-order term."""
        # Construct index list for upper triangle gather operation
        ut = torch.unbind(torch.triu_indices(
            self.U.shape[-1], self.U.shape[-1], 1))
        distance = self.distance[..., ut[0], ut[1]]

        # deal with single and batch problem
        U = self.U.unsqueeze(0) if self.U.dim() == 1 else self.U

        # build the whole gamma, shortgamma (without 1/R) and triangular gamma
        gamma = torch.zeros(*U.shape, U.shape[-1])
        gamma_tr = torch.zeros(U.shape[0], len(ut[0]))

        # diagonal values is so called chemical hardness Hubbert
        if self.gamma_type == 'slater':
            gamma.diagonal(0, -(U.dim() - 1))[:] = U
        elif self.gamma_type == 'slater_short':
            gamma.diagonal(0, -(U.dim() - 1))[:] = -U

        alpha, beta = U[..., ut[0]] * 3.2, U[..., ut[1]] * 3.2

        # mask of homo or hetero Hubbert in triangular gamma
        mask_homo, mask_hetero = alpha == beta, alpha != beta
        mask_homo[distance.eq(0)], mask_hetero[distance.eq(0)] = False, False
        r_homo, r_hetero = 1. / distance[mask_homo], 1. / distance[mask_hetero]

        # homo Hubbert
        aa, dd_homo = alpha[mask_homo], distance[mask_homo]
        taur = aa * dd_homo
        efac = torch.exp(-taur) / 48. * r_homo
        gamma_tr[mask_homo] = \
            (48. + 33. * taur + 9. * taur ** 2 + taur ** 3) * efac
        if self.gamma_type == 'slater':  # -> add 1 / distances
            gamma_tr[mask_homo] = \
                r_homo - gamma_tr[mask_homo]

        # hetero Hubbert
        aa, bb = alpha[mask_hetero], beta[mask_hetero]
        dd_hetero = distance[mask_hetero]
        aa2, aa4, aa6 = aa ** 2, aa ** 4, aa ** 6
        bb2, bb4, bb6 = bb ** 2, bb ** 4, bb ** 6
        rab, rba = 1 / (aa2 - bb2), 1 / (bb2 - aa2)
        exp_a, exp_b = torch.exp(-aa * dd_hetero), torch.exp(-bb * dd_hetero)
        val_ab = exp_a * (0.5 * aa * bb4 * rab ** 2 -
                          (bb6 - 3. * aa2 * bb4) * rab ** 3 * r_hetero)
        val_ba = exp_b * (0.5 * bb * aa4 * rba ** 2 -
                          (aa6 - 3. * bb2 * aa4) * rba ** 3 * r_hetero)
        gamma_tr[mask_hetero] = val_ab + val_ba

        if self.gamma_type == 'slater':  # -> add 1 / distances
            gamma_tr[mask_hetero] = (r_hetero - gamma_tr[mask_hetero])

        # return symmetric gamma values
        gamma[..., ut[0], ut[1]] = gamma_tr
        gamma[..., ut[1], ut[0]] = gamma[..., ut[0], ut[1]]
        return gamma

    def gaussian(self):
        """Build the Gaussian type gamma in second-order term."""
        raise NotImplementedError('Not implement gaussian yet.')


def fermi(eigenvalue: Tensor, nelectron: Tensor, kT=0.):
    """Fermi-Dirac distributions without smearing.

    Arguments:
        eigenvalue: Eigen-energies.
        nelectron: number of electrons.
        kT: temperature.

    Returns
        occ: occupancies of electrons
    """
    # make sure each system has at least one electron
    assert False not in torch.ge(nelectron, 1)
    # the number of full occupied state
    electron_pair = torch.true_divide(nelectron.clone().detach(), 2).int()
    # the left single electron
    electron_single = (nelectron.clone().detach() % 2).unsqueeze(1)

    # zero temperature
    if kT != 0:
        raise NotImplementedError('not implement smearing method.')

    # occupied state for batch, if full occupied, occupied will be 2
    # with unpaired electron, return 1
    occ_ = pack([
        torch.cat((torch.ones(electron_pair[i]) * 2, electron_single[i]), 0)
        for i in range(nelectron.shape[0])])

    # pad the rest unoccupied states with 0
    occ = F.pad(input=occ_, pad=(
        0, eigenvalue.shape[-1] - occ_.shape[-1]), value=0)

    # all occupied states (include full and not full occupied)
    nocc = (nelectron.clone().detach() / 2.).ceil()

    return occ, nocc
