"""Physical properties routine."""
import torch
import numpy as np
from numbers import Real
from typing import Optional, Tuple, Union, Iterable, Any
from tbmalt.common.batch import pack
Tensor = torch.Tensor
ndarray = np.ndarray


class Properties:

    def __init__(self, properties: list, system: object, qzero: Tensor,
                 charge: Tensor, overlap: Tensor, density: Tensor, **kwargs):
        self.qzero = qzero
        self.system = system
        self.charge = charge

    @property
    def dipole(self) -> Tensor:
        """Return dipole moments."""
        return torch.sum((self.qzero - self.charge).unsqueeze(-1) *
                         self.system.positions, 1)

    @property
    def mulliken_charge(self, overlap: Tensor, density: Tensor,
                        atom_orbitals=None, **kwargs) -> Tensor:
        """Calculate Mulliken charge for both batch system.

        Arguments:
            overlap: 2D or 3D overlap matrix.
            density: 2D or 3D density matrix.
            atom_orbitals: Orbital number for each atom, which is (l + 1) ** 2.

        """
        orbital_resolve = kwargs.get('orbital_resolve', False)

        # sum overlap and density by hadamard product, get charge by orbital
        charge_orbital = (density * overlap).sum(dim=1)

        if orbital_resolve:

            # return orbital resolved charge
            return charge_orbital
        else:
            assert atom_orbitals is not None

            # get a list of accumulative orbital indices
            ind_cumsum = [torch.cat((
                torch.zeros(1), torch.cumsum(ii[ii.ne(0)], -1))).long()
                for ii in atom_orbitals]

            # return charge of each atom for batch system
            return pack([torch.stack([(icharge[ii: jj]).sum() for ii, jj in zip(
                icumsum[:-1], icumsum[1:])])
                   for icumsum, icharge in zip(ind_cumsum, charge_orbital)])

    @property
    def homo_lumo(self):
        pass

    @property
    def gap(self):
        pass

    @property
    def net_onsite(self):
        pass

    @property
    def polarizability(self):
        pass

    @property
    def pdos(self):
        pass

    @property
    def dos(self):
        pass

    @property
    def band_structure(self):
        pass

def mulliken(overlap: Tensor, density: Tensor, atom_orbitals=None,
             **kwargs) -> Tensor:
    """Calculate Mulliken charge for both batch system.

    Arguments:
        overlap: 2D or 3D overlap matrix.
        density: 2D or 3D density matrix.
        atom_orbitals: Orbital number for each atom, which is (l + 1) ** 2.

    """
    orbital_resolve = kwargs.get('orbital_resolve', False)

    # sum overlap and density by hadamard product, get charge by orbital
    charge_orbital = (density * overlap).sum(dim=1)

    if orbital_resolve:

        # return orbital resolved charge
        return charge_orbital
    else:
        assert atom_orbitals is not None

        # get a list of accumulative orbital indices
        ind_cumsum = [torch.cat((
            torch.zeros(1), torch.cumsum(ii[ii.ne(0)], -1))).long()
            for ii in atom_orbitals]

        # return charge of each atom for batch system
        return pack([torch.stack([(icharge[ii: jj]).sum() for ii, jj in zip(
            icumsum[:-1], icumsum[1:])])
            for icumsum, icharge in zip(ind_cumsum, charge_orbital)])


def _gaussian_broadening(energy: Union[Tensor, Real],
                         eps: Union[Tensor, Real], sigma: Real) -> Tensor:
    """Gaussian broadening factor used when calculating the PDoS.


    Arguments:
        energy: Energy(s)
        eps: Eigenvalue(s)
        sigma: Broadening factor.

    Returns:
        e_broadened: Broadened energy value.
    """

    return torch.erf((energy - eps) / (np.sqrt(2) * sigma))


def dos(eps: Tensor, energies: Tensor, sigma: Real = 0.0,
        offset: Optional[Real] = None, mask: Optional[Tensor] = None,
        scale: bool = False) -> Tensor:
    r"""Calculate the density of states.

    This calculates and returns the Density of States (DoS). If desired the
    states included in the DoS an be controlled via the ``mask`` argument.

    Arguments:
        eps: Eigenvalues.
        energies: Energy values to evaluate the PDoS at. These are assumed to
            be relative to the ``offset`` value, if it is specified.
        sigma: Smearing width for gaussian broadening function. [DEFAULT=0]
        offset: Indicates that ``energies`` are given with respect to a offset
            value, e.g. the fermi energy.
        mask: The ``mask`` option can be used to control which states are used
            when constructing the PDoS. Only states that are unmasked (True)
            will be used, all others will be ignored.
        scale: Scales the DoS to have a maximum value of 1.

    Returns:
        dos: The densities of states.

    Notes:
        The DoS is calculated via an equation equivalent to:
        .. math::

            g(E)= C_{\mu,i} \delta(E-\epsilon_{i})

        Where g(E) is the density of states at an energy value E, and δ(E-ε)
        is the smearing width calculated as:
        .. math::

            \delta(E-\epsilon) = \frac{\frac{E - \epsilon + \frac{\Delta E}{2}}{\sqrt{2} \sigma)}
            - \frac{E - \epsilon - \frac{\Delta E}{2}}{\sqrt{2} \sigma)}}{2 \Delta E}

        Where ΔE is the difference in energy between neighbouring points.

        It may be useful, such as in the creation of a cost function, to have
        only specific states (i.e. HOMO, HOMO-1, etc.) used to construct the
        PDoS. State selection can be achieved via the ``mask`` argument. This
        should be a boolean tensor with a shape matching that of ``eps``. Only
        states whose mask value is True will be included in the DoS, e.g.;
            mask = torch.Tensor([True, False, False, False])
        would only use the first state when constructing the DoS.

    """
    # Mask out selected eigen-states, if instructed to do so
    if mask is not None:
        eps = eps[mask]

    if offset is not None:  # Apply the offset, if applicable.
        eps = eps - offset

    # Construct gaussian smeared DoS terms. This is analogous to how FHI-aims
    # does it.
    de = energies[1] - energies[0]
    ga = _gaussian_broadening((energies - (de / 2)).unsqueeze(1), eps, sigma)
    gb = _gaussian_broadening((energies + (de / 2)).unsqueeze(1), eps, sigma)
    g = (gb - ga) / (2.0 * de)

    # Compute the densities of states
    distribution = torch.sum(g, -1)

    # Rescale the DoS so that it has a maximum peak value of 1.
    if scale:
        distribution = distribution / distribution.sum(-2)

    return distribution


def pdos(C: Tensor, S: Tensor, eps: Tensor, energies: Tensor,
         sigma: Real = 0.1, offset: Optional[Real] = None,
         resolve_by: Optional[Iterable[Any]] = None,
         mask: Optional[Tensor] = None,
         scale: bool = False,
         ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Calculate the projected density of states.

    This calculates and returns the Projected Density of States (PDoS). By
    default a single distribution is returned for each basis. However, the
    ``resolve_by`` option can be used to resolve the PDoS by atomic number,
    atom index, etc. Furthermore, the states included in the PDoS can be
    controlled via the ``mask`` argument.

    Arguments:
        C: Coefficient matrix with eigenvectors represented as columns.
        S: Overlap matrix.
        eps: Eigenvalues.
        energies: Energy values to evaluate the PDoS at. These are assumed to
            be relative to the ``offset`` value, if it is specified.
        sigma: Smearing width for gaussian broadening function. [DEFAULT=0.0]
        offset: Indicates that ``energies`` are given with respect to a offset
            value, e.g. the fermi energy.
        resolve_by: By default, one distribution is returned for each basis.
            However, distributions can be aggregated by specific features such
            as atomic number, through the use of the ``resolve_by`` option, if
            required. Each row of the ``resolve_by`` array should correspond
            to a basis and each column to a property.
        mask: The ``mask`` option can be used to control which states are used
            when constructing the PDoS. Only states that are unmasked (True)
            will be used, all others will be ignored.
        scale: scales the distributions so that the maximum value of the sum
            of the distributions (DoS) is equal to 1.

    Returns:
        pdos: The projected densities of states.
        labels: Labels for the resolved distributions. Only returned when
            ``resolve_by`` is specified.

    Notes:
        The PDoS is calculated via an equation equivalent to:
        .. math::

            g(\mu, E)=\sum_i \sum_v C_{\nu,i}^* S_{\mu v} C_{\mu,i} \delta(E-\epsilon_{i})

        Where g(μ, E) is the density of states of orbital μ at an energy value
        E and δ(E-ε) is the smearing width calculated as:
        .. math::

            \delta(E-\epsilon) = \frac{\frac{E - \epsilon + \frac{\Delta E}{2}}{\sqrt{2} \sigma)}
            - \frac{E - \epsilon - \frac{\Delta E}{2}}{\sqrt{2} \sigma)}}{2 \Delta E}

        Where ΔE is the difference in energy between neighbouring points.

        It may be useful, such as in the creation of a cost function, to have
        only specific states (i.e. HOMO, HOMO-1, etc.) used to construct the
        PDoS. State selection can be achieved via the ``mask`` argument. This
        should be a boolean tensor with a shape matching that of ``eps``. Only
        states whose mask value is True will be included in the PDoS, e.g.;
            mask = torch.Tensor([True, False, False, False])
        would only use the first state when constructing the PDoS.

        If desired, the PDoS can be resolved by atomic number, or another
        arbitrary property, through the use of the ``resolve_by`` argument.
        When ``resolve_by`` is passed, this function will associate each
        distribution with an entry in ``resolve_by``. It will then take all
        distributions with matching entries in ``resolve_by`` and sum them
        together. For example:
            resolve_by = ['H', 'C', 'H', 'H', 'O', 'H']
        will sum distributions 0, 2, 3 & 5 together, resulting in a single
        distribution being returned for each element. Resolution can be based
        on multiple properties if desired.

    """
    # Mask out selected eigen-states, if instructed to do so
    if mask is not None:
        eps, C = eps[mask], C[:, mask]

    if offset is not None:  # Apply the offset, if applicable.
        eps = eps - offset

    # Construct gaussian smeared DoS terms
    de = energies[1] - energies[0]
    ga = _gaussian_broadening((energies - (de / 2)).unsqueeze(1), eps, sigma)
    gb = _gaussian_broadening((energies + (de / 2)).unsqueeze(1), eps, sigma)
    g = (gb - ga) / (2.0 * de)

    # Compute the projected densities of states
    distributions = torch.einsum('vi,ui,vu,ei->ue', C, C, S, g)

    # Rescale the PDoS distributions so that the DoS has a maximum value of 1.
    if scale:
        distributions = distributions / distributions.sum(-2)

    # If instructed to resolve the projected density of states
    if resolve_by is not None:
        # Convert Tensors to np.arrays; np.unique must be used as torch.unique
        # fails on multi-column systems.
        if isinstance(resolve_by, Tensor):
            resolve_by = resolve_by.detach().numpy()

        # Identify all unique labels & the indices associated with them
        label_u, ind_u = np.unique(resolve_by, axis=0, return_inverse=True)
        # Loop over unique types & aggregate distributions of that type via summation
        distributions = torch.stack(
            [torch.sum(distributions[np.where(ind_u == i)[0]], 0)
             for i in range(len(label_u))])

        # Return the, now resolved, distributions along with the labels.
        return distributions, label_u

    else:  # If not resolving, return the distributions as is.
        return distributions


def band_pass_states(eps: Tensor, n_homo: int, n_lumo: int,
                     fermi_energy: Real) -> Tensor:
    """Creates a mask which masks out states too far from the fermi level.

    Masks out all but the first n_homo & n_lumo energy levels below &
    above the fermi level respectively.

    Arguments:
        eps: The eigenvalues.
        n_homo: Number of states below the fermi energy to include.
        n_lumo: Number of states above the fermi energy to include.
        fermi_energy: The fermi energy value.

    Returns:
        mask: A boolean mask which is True for selected states.

    Raises:
        ValueError: Raised if n_homo/n_lumo exceeds the number of available
            homo/lumo states.
    """
    # State indices relative to the "HOMO", e.g:
    #   [-n, ..., -2, -1, 0, 1, 2, ..., +n]
    # Where zero resides at the location of the "HOMO" and 1 at the "LUMO"
    li = (torch.arange(len(eps)) - torch.where(eps <= fermi_energy)[0].max())

    # Ensure n_homo/lumo are not too large for the number of available states
    homo_count = len(torch.where(li <= 0)[0])
    lumo_count = (len(li) - homo_count)
    if n_homo > homo_count:
        raise ValueError(
            f'Too many HOMO levels requested, only {homo_count} available.')
    if n_lumo > lumo_count:
        raise ValueError(
            f'Too many LUMO levels requested, only {lumo_count} available.')

    # Mask out states outside of the band filter
    mask = (-n_homo < li) & (li <= n_lumo)

    # Return the mask
    return mask
