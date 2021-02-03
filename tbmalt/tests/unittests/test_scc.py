"""Test SCC DFTB."""
import torch
import os
import pytest
from torch.autograd import gradcheck
from ase.build import molecule as molecule_database
from tbmalt.common.structures.system import System
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.tb.sk import SKT
from tbmalt.tb.dftb.scc import Scc
from tbmalt.common.parameter import Parameter
torch.set_default_dtype(torch.float64)


os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')


def test_nonscc_ase_single():
    """Test non-SCC DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('H2O')])
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter)
    print('charge single', scc.charge)


def test_nonscc_ase():
    """Test non-SCC DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('CH4'),
                                      molecule_database('H2O'),
                                      molecule_database('CH3CH2O')])
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter)
    print('charge batch', scc.charge, scc.qatom)


def test_scc_property():
    """Test non-SCC DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('CH4'),
                                      molecule_database('H2'),
                                      molecule_database('CH3CH2O')])
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    properties = ['dipole']
    scc = Scc(molecule, skt, parameter, properties)
    print('charge', scc.charge, 'dipole', scc.properties.dipole)


def test_read_compr_single(device):
    """Test SKF data with various compression radii."""
    molecule = System.from_ase_atoms([molecule_database('CH4')])
    compression_radii_grid = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
    compression_r = torch.tensor([[3., 3.1, 3.2, 3.3, 3.4, 0, 0, 0]])
    sk = IntegralGenerator.from_dir(
        './slko/compr', molecule, repulsive=True,
        sk_type='compression_radii', homo=False, interpolation='BicubInterp',
        compression_radii_grid=compression_radii_grid)

    skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
              fix_U=True)
    parameter = Parameter()
    properties = ['dipole']
    scc = Scc(molecule, skt, parameter, properties)
    print('charge', scc.charge, 'dipole', scc.properties.dipole)


def test_read_compr_batch(device):
    """Test SKF data with various compression radii."""
    molecule = System.from_ase_atoms([molecule_database('CH4'),
                                      molecule_database('NH3')])
                                      # molecule_database('C2H6')])
                                      # molecule_database('CH3CH2O')])
    compression_radii_grid = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
    compression_r = torch.tensor([[3., 3.1, 3.2, 3.3, 3.4, 0, 0, 0],
                                  [3., 3., 3., 3., 0., 0., 0., 0]])
                                  # [3., 3., 3., 3., 3., 3.5, 3.5, 4.]])
                                  # [3., 3., 3., 3., 3., 3.5, 3.5, 4.]])
    sk = IntegralGenerator.from_dir(
        './slko/compr', molecule, repulsive=True,
        sk_type='compression_radii', homo=False, interpolation='BicubInterp',
        compression_radii_grid=compression_radii_grid)

    skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
              fix_U=True)
    parameter = Parameter()
    properties = ['dipole']
    scc = Scc(molecule, skt, parameter, properties)
    print('charge', scc.charge, 'dipole', scc.properties.dipole)
