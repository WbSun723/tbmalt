"""To write skf files used by DFTB+."""
import numpy as np
import torch
from tbmalt.common.structures.system import System
from tbmalt.io.loadskf import IntegralGenerator
torch.set_default_dtype(torch.float64)

# Path to the original skf file
path_to_skf = './siband.hdf'

# Number of grid points to write
start_point = 0.2
end_point = 18.0
grid_dist = 0.02
N = int((end_point - start_point) / grid_dist + 1)

def generate_grid(start_point, end_point, N):
    """Obtain the integral values of grid points for sk table."""
    # An example system including Si-Si interaction to initialize the program
    latvec = torch.tensor([[5.8, 0., 0.], [0., 5.8, 0.], [0., 0., 5.8]])
    positions = torch.tensor([
        [0.0, 0.0, 0.0], [1.45, 1.45, 1.45], [1.45, 4.35, 4.35], [4.35, 1.45, 4.35],
        [4.35, 4.35, 1.45], [2.9	, 2.9, 0.0], [2.9, 0.0, 2.9], [0.0, 2.9, 2.9]])
    numbers = torch.tensor([14, 14, 14, 14, 14, 14, 14, 14])
    system = System(numbers, positions, latvec)

    # Read the optimized spline parameters
    with open("./abcd/abcd.txt", "r") as f:
        abcd = eval(f.read())

    # Build integral generator
    sk_pred = IntegralGenerator.from_dir(
            path_to_skf, system, repulsive=False,
            interpolation='spline', sk_type='h5py', pred=abcd)

    # Atom pair to write
    atom_pair = torch.tensor([14, 14])

    # Generate a grid list
    distance = torch.linspace(start_point, end_point, N)

    # Integral values for H
    Hss = sk_pred(distance, atom_pair, torch.tensor([0, 0]), hs_type='H', get_abcd='abcd')
    Hsp = sk_pred(distance, atom_pair, torch.tensor([0, 1]), hs_type='H', get_abcd='abcd')
    Hsd = sk_pred(distance, atom_pair, torch.tensor([0, 2]), hs_type='H', get_abcd='abcd')
    Hpp = sk_pred(distance, atom_pair, torch.tensor([1, 1]), hs_type='H', get_abcd='abcd')
    Hpd = sk_pred(distance, atom_pair, torch.tensor([1, 2]), hs_type='H', get_abcd='abcd')
    Hdd = sk_pred(distance, atom_pair, torch.tensor([2, 2]), hs_type='H', get_abcd='abcd')
    H = torch.cat((Hdd, Hpd, Hpp, Hsd, Hsp, Hss), 1)

    # Integral values for S
    Sss = sk_pred(distance, atom_pair, torch.tensor([0, 0]), hs_type='S', get_abcd='abcd')
    Ssp = sk_pred(distance, atom_pair, torch.tensor([0, 1]), hs_type='S', get_abcd='abcd')
    Ssd = sk_pred(distance, atom_pair, torch.tensor([0, 2]), hs_type='S', get_abcd='abcd')
    Spp = sk_pred(distance, atom_pair, torch.tensor([1, 1]), hs_type='S', get_abcd='abcd')
    Spd = sk_pred(distance, atom_pair, torch.tensor([1, 2]), hs_type='S', get_abcd='abcd')
    Sdd = sk_pred(distance, atom_pair, torch.tensor([2, 2]), hs_type='S', get_abcd='abcd')
    S = torch.cat((Sdd, Spd, Spp, Ssd, Ssp, Sss), 1)

    # Combined table
    HS = torch.cat((H, S), 1)

    return HS

def write_skf(start_point, end_point, grid_dist, N):
    """Write standard skf file."""
    # Obtain integral values
    HS = generate_grid(start_point, end_point, N)

    # Parameters for skf file
    N_zeros = int((start_point - grid_dist) / grid_dist)
    zeros = torch.zeros(N_zeros, 10)
    ones = torch.ones(N_zeros, 10)

    # Add the values when interactions are too close
    HS_short = torch.cat((zeros, ones), 1)

    # Complete table
    HS_total = torch.cat((HS_short, HS), 0)

    # Read the original skf and obtain related information
    with open('./standard.skf', 'r') as file:
        ori_lines = file.readlines()

    # Write the new skf file containing ML-features
    with open('./Si-Si.skf', 'a') as file:
        file.write(str(grid_dist) + " " + str(N - 1))
        file.write("\n")
        file.writelines(ori_lines[1: 3])

    f = open('./Si-Si.skf', 'a')
    np.savetxt(f, HS_total)
    f.close()

    # The spline parameters are copied from the original skf file
    with open('./Si-Si.skf', 'a') as file:
        file.writelines(ori_lines[94: 110])
        file.write("\n")
        file.write("This SPLINE is just a DUMMY-SPLINE!!!!!!!!!!!!!!!")
        file.write("\n")
        file.write("\n")

write_skf(start_point, end_point, grid_dist, N)
