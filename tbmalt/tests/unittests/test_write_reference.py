"""Write SKF files to binary files."""
import os
import pytest
import torch
from tbmalt.utils.reference.write_reference import CalReference
from tbmalt.io.loadhdf import LoadHdf
from tbmalt.common.structures.system import System
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')
os.system('cp -r /home/gz_fan/Public/tbmalt/aims .')
os.system('cp -r /home/gz_fan/Public/tbmalt/dftbplus .')
os.system('cp -r /home/gz_fan/Public/tbmalt/dataset .')
_path = os.getcwd()


def test_aims_results_to_hdf():
    """Test writing h5py binary file from FHI-aims calculations."""
    # -> define all input parameters
    properties = ['dipole', 'charge', 'homo_lumo', 'energy',
                  'formation_energy', 'alpha_mbd',
                  'hirshfeld_volume', 'hirshfeld_volume_ratio']
    path_to_input = './dataset/ani_gdb_s03.h5'
    input_type = 'ANI-1'
    reference_size = 2000
    reference_type = 'aims'
    output_name = 'aims03.hdf'
    path_to_aims_specie = os.path.join(_path, 'aims/species_defaults/tight')
    path_to_aims = os.path.join(_path, 'aims/aims.x')

    w_aims_ani1 = CalReference(path_to_input, input_type,
                               reference_size, reference_type,
                               path_to_aims_specie=path_to_aims_specie,
                               path_to_aims=path_to_aims)

    # calculate properties
    results = w_aims_ani1(properties)

    # write results (properties) to hdf
    CalReference.to_hdf(results, w_aims_ani1, properties, mode='w',
                        output_name=output_name)

    # test the hdf reference
    numbers, positions, data = LoadHdf.load_reference(
        output_name, reference_size, properties)

    # make sure the data type consistency
    print('numbers', w_aims_ani1.numbers)
    print('dipole', results['dipole'])
    LoadHdf.get_info(output_name)  # return dataset information


# def test_dftbplus_nonscc_results_to_hdf():
#     """Test repulsive of hdf."""
#     # -> define all input parameters
#     properties = ['dipole', 'charge', 'homo_lumo', 'energy',
#                   'formation_energy']
#     path_to_input = './dataset/ani_gdb_s02.h5'
#     input_type = 'ANI-1'
#     reference_size = 6000
#     reference_type = 'dftbplus'
#     path_to_dftbplus = os.path.join(_path, 'dftbplus/dftb+')
#     path_to_skf = os.path.join(_path, './slko/init')

#     w_dftb_ani1 = CalReference(path_to_input, input_type,
#                                reference_size, reference_type,
#                                path_to_skf=path_to_skf,
#                                path_to_dftbplus=path_to_dftbplus)

#     # calculate properties
#     output_name = 'nonscc.hdf'
#     results = w_dftb_ani1(properties, dftb_type='nonscc')

#     # write results (properties) to hdf
#     CalReference.to_hdf(results, w_dftb_ani1, properties, mode='w',
#                         output_name=output_name)

#     # load the the generated dataset
#     numbers, positions, data = LoadHdf.load_reference(
#         output_name, reference_size, properties)

#     # make sure the data type consistency
#     print('numbers', w_dftb_ani1.numbers)
#     print('dipole', results['dipole'])
#     LoadHdf.get_info(output_name)  # return dataset information


# def test_dftbplus_scc_results_to_hdf():
#     """Test repulsive of hdf."""
#     # -> define all input parameters
#     properties = ['dipole', 'charge', 'homo_lumo', 'energy',
#                   'formation_energy']
#     path_to_input = './dataset/ani_gdb_s02.h5'
#     input_type = 'ANI-1'
#     reference_size = 6000
#     reference_type = 'dftbplus'
#     path_to_dftbplus = os.path.join(_path, 'dftbplus/dftb+')
#     path_to_skf = os.path.join(_path, './slko/init')

#     w_dftb_ani1 = CalReference(path_to_input, input_type,
#                                reference_size, reference_type,
#                                path_to_skf=path_to_skf,
#                                path_to_dftbplus=path_to_dftbplus)

#     # calculate properties
#     output_name = 'scc.hdf'
#     results = w_dftb_ani1(properties, dftb_type='scc')

#     # write results (properties) to hdf
#     CalReference.to_hdf(results, w_dftb_ani1, properties, mode='w',
#                         output_name=output_name)

#     # load the the generated dataset
#     numbers, positions, data = LoadHdf.load_reference(
#         output_name, reference_size, properties)

#     # make sure the data type consistency
#     print('numbers', w_dftb_ani1.numbers)
#     print('dipole', results['dipole'])
#     LoadHdf.get_info(output_name)  # return dataset information
