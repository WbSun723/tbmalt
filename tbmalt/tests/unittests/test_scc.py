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
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)


os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')


def test_ase_h2o():
    """Test H2O DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('H2O')])
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    parameter.scc = 'nonscc'
    scc = Scc(molecule, skt, parameter)
    assert torch.max(abs(scc.properties.charge - torch.tensor([
        6.7551958288882625, 0.6224020855558695, 0.6224020855558694]))) < 1E-14

    parameter.scc = 'scc'
    scc = Scc(molecule, skt, parameter)
    assert torch.max(abs(scc.properties.charge - torch.tensor([
        6.5855898436436817, 0.7072050781781599, 0.7072050781781598]))) < 1E-12


def test_ase_c2h6():
    """Test C2H6 DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('C2H6')])
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter)
    assert torch.max(abs(scc.properties.charge - torch.tensor([
        4.1916322572147706, 4.1916322572147635, 0.9361224815474558,
        0.9361226306188899, 0.9361226306188895, 0.9361224815474540,
        0.9361226306188882, 0.9361226306188880]))) < 1E-14


def test_batch_ase():
    """Test batch DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('CH4'),
                                      molecule_database('H2O'),
                                      molecule_database('C2H6')])
    sktable = IntegralGenerator.from_dir('./slko/auorg-1-1', molecule)
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter)
    assert torch.max(abs(scc.properties.charge - torch.tensor([
        [4.3053789406571052, 0.9236552648357237, 0.9236552648357236,
         0.9236552648357227, 0.9236552648357238, 0., 0., 0.],
        [6.5855898436436817, 0.7072050781781599, 0.7072050781781598,
         0., 0., 0., 0., 0.],
        [4.1916322572147706, 4.1916322572147635, 0.9361224815474558,
         0.9361226306188899, 0.9361226306188895, 0.9361224815474540,
         0.9361226306188882, 0.9361226306188880]]))) < 1E-10


def test_scc_spline():
    """Test non-SCC DFTB from ase input."""
    molecule = System.from_ase_atoms([molecule_database('CH4')])
    sktable = IntegralGenerator.from_dir(
        './slko/auorg-1-1', molecule, interpolation='spline')
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    properties = ['dipole']
    scc = Scc(molecule, skt, parameter, properties)
    sktable = IntegralGenerator.from_dir(
        './slko/auorg-1-1', molecule, repulsive=False, interpolation='spline', with_variable=True)
    skt = SKT(molecule, sktable, with_variable=True)
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
        sk_type='compression_radii', homo=False, interpolation='bicubic_interpolation',
        compression_radii_grid=compression_radii_grid)

    skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
              fix_U=True)
    parameter = Parameter()
    properties = ['dipole']
    scc = Scc(molecule, skt, parameter, properties)


def test_read_compr_batch(device):
    """Test SKF data with various compression radii."""
    molecule = System.from_ase_atoms([molecule_database('CH4'),
                                      molecule_database('NH3'),
                                      molecule_database('C2H6')])
    compression_radii_grid = torch.tensor([
        01.00, 01.50, 02.00, 02.50, 03.00, 03.50, 04.00,
        04.50, 05.00, 05.50, 06.00, 07.00, 08.00, 09.00, 10.00])
    compression_r = torch.tensor([[3., 3.1, 3.2, 3.3, 3.4, 0, 0, 0],
                                  [3., 3., 3., 3., 0., 0., 0., 0],
                                  [3., 3., 3., 3., 3., 3.5, 3.5, 3.5]])
    sk = IntegralGenerator.from_dir(
        './slko/compr', molecule, repulsive=True,
        sk_type='compression_radii', homo=False, interpolation='bicubic_interpolation',
        compression_radii_grid=compression_radii_grid)

    skt = SKT(molecule, sk, compression_radii=compression_r, fix_onsite=True,
              fix_U=True)
    parameter = Parameter()
    properties = ['dipole']
    scc = Scc(molecule, skt, parameter, properties)
