import numpy as np
import matplotlib.pyplot as plt
import torch
from tbmalt.physics.properties import *


# When we compare minimal to non-minimal basis DFT calculations we see
#   that the minimal system's PDoS is effectively squeezed. Would this
#   be problematic if we tried to force the a minimal system to "spread-out"?
#   perhaps split up the "character" from the span.
# Must give an example showing exactly how mask is used.

def load_CH4(set_number=0):
    """Returns DFT level PDoS test data for CH4 molecule."""
    # eV2Ha = 1 / 27.211386
    tt = torch.tensor
    # __SETUP__
    set_lookup = {0: 'CH4_Minimal',
                  1: 'CH4',
                  2: 'CH4_2',
                  3: 'CH4_3'}


    path = f'/home/ajmhpc/Projects/Cormorant/cormorant/testing/data/PDoS/{set_lookup[set_number]}/'


    geometry_file = f'{path}geometry.in'
    basis_set_file = f'{path}basis-indices.out'
    eigen_data_file = f'{path}mos.aims'
    aims_out_file = f'{path}aims.out'
    overlap_file = f'{path}overlap-matrix.out'
    overlap_file_2 = f'{path}omat.aims'

    # __IO__
    # Read in the geometry file
    system = read(geometry_file)

    # Read in the eigenvalues and eigenvectors
    states, C = aims_io.read_kohn_sham_eigendata(eigen_data_file)

    # Read the fermi energy value.
    fermi_energy = aims_io.read_fermi(aims_out_file)

    # Read in the basis set, this is a structured numpy array
    bases = aims_io.read_basis_indices(basis_set_file)

    # "basis" is missing atomic number; so we must add it ourselves
    z_list = system.get_atomic_numbers()[bases['a'] - 1]
    bases = recfunctions.append_fields(bases, 'z', z_list,
                                       dtypes=['i4'], usemask=False)

    # Overlap matrix
    try:
        S = aims_io.read_overlap_matrix(overlap_file)
    except FileNotFoundError:
        S = aims_io.read_overlap_matrix(overlap_file_2)

    # Get the DoS and PDoS's
    dos = {
        'total': aims_io.read_dos(f'{path}KS_DOS_total_raw.dat'),
        'H': aims_io.read_dos(f'{path}H_l_proj_dos_raw.dat'),
        'C': aims_io.read_dos(f'{path}C_l_proj_dos_raw.dat')
    }

    return tt(S), tt(C), tt(states), fermi_energy, bases, dos


if __name__ == '__main__':
    # When using the error function version of the gaussian we get
    # v = 13.60569 consistently. This is the The Rydberg constant.
    import sys
    sys.path.append('/home/ajmhpc/Projects/DFTBMaLT')
    from dftbmalt.io import aims_io
    from ase.io import read
    from numpy.lib import recfunctions
    import matplotlib.pyplot as plt

    tt = torch.tensor

    sigma = 0.1 / 27.211386
    S, C, states, fermi_energy, bases, dos = load_CH4(1)
    energy_values = tt(dos['total']['energy'])


    # __MASKING__

    # The number of energy levels below and above the fermi level to keep
    n_homo, n_lumo = 5, 4
    eps_mask = band_pass_states(states, n_homo, n_lumo, fermi_energy)


    S2, C2, states2, fermi_energy2,  bases2, dos2 = load_CH4(0)
    ref_1 = pdos(C, S, states, energy_values, sigma=sigma, mask=eps_mask)
    # ref_x = pdos(C2, S2, states2, energy_values, sigma=sigma)
    #

    #
    # calc_1 = pdos(C, S, states, energy_values, sigma=sigma, eps_mask=eps_mask)

    #
    #
    # ref_2 = dos2['total']['total']
    # calc_dos = dos_func(C, states, energy_values, sigma=sigma)
    # plt.plot(energy_values, ref_1.sum(0) / ref_1.sum(0).max(), 'k-')
    # plt.plot(energy_values, calc_dos / calc_dos.max(), 'r:')
    # plt.show()

    # plt.plot(energy_values, ref_calc.sum(0)/ref_calc.sum(0).max(), 'k-')
    # plt.plot(energy_values, calc.sum(0)/calc.sum(0).max(), 'r:')
    # plt.plot(energy_values, dos2['total']['total'] / dos2['total']['total'].max(), 'g:')
    # plt.show()

    # plt.plot(energy_values, ref_calc.sum(0)/ref_calc.sum(0).sum(), 'k-')
    # plt.plot(energy_values, calc.sum(0)/calc.sum(0).sum(), 'r:')
    # plt.show()

    # ref_norm = dos['total']['total'].max()
    # ref = dos['total']['total'] / ref_norm
    # calc, label_u = pdos(C, S, states, energy_values, sigma=sigma, resolve_by=bases['z'])
    # calc = calc / calc.sum(0).max()
    #
    # ref_c = dos['C']['1'] / ref_norm
    # calc_c = calc[1]
    #
    # plt.plot(energy_values, ref, 'k-')
    # plt.plot(energy_values, calc.sum(0), 'r:')
    # plt.show()


    _ = ...
