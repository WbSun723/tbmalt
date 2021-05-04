"""Test SCC DFTB."""
import torch
import os
import pytest
from tbmalt.common.structures.system import System
from tbmalt.io.loadskf import IntegralGenerator
from tbmalt.tb.sk import SKT
from tbmalt.tb.dftb.scc import Scc
from tbmalt.common.parameter import Parameter
from tbmalt.common.structures.periodic import Periodic
from tbmalt.tb.coulomb import Coulomb
torch.set_printoptions(15)
torch.set_default_dtype(torch.float64)


os.system('cp -r /home/gz_fan/Public/tbmalt/slko .')


def test_scc_ch4_npe():
    """Test SCC DFTB for ch4 without periodic boundary condition."""
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]])
    numbers = torch.tensor([6, 1, 1, 1, 1])
    molecule = System(numbers, positions)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    skt = SKT(molecule, sktable)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter)
    assert torch.max(abs(scc.charge - torch.tensor([
        4.6010835947475499, 0.84036067839669026, 0.85285190895192031,
        0.85285190895192053, 0.85285190895192087]))) < 1E-10, 'Tolerance check'


def test_scc_ch4_pe():
    """Test SCC DFTB for ch4 with periodic boundary condition."""
    latvec = torch.tensor([[4., 4., 0.], [0., 5., 0.], [0., 0., 6.]])
    cutoff = torch.tensor([9.98])
    positions = torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]])
    numbers = torch.tensor([6, 1, 1, 1, 1])
    molecule = System(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    periodic = Periodic(molecule, molecule.cell, cutoff=cutoff)
    skt = SKT(molecule, sktable, periodic)
    coulomb = Coulomb(molecule, periodic)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter, coulomb, periodic)
    assert torch.max(abs(scc.charge - torch.tensor([
        4.6186492039337015, 0.83091046864033347, 0.85337011280373809,
        0.85040755555711756, 0.84666265906510807]))) < 1E-8, 'Tolerance check'


def test_scc_c2h6_pe():
    """Test SCC DFTB for c2h6 with periodic boundary condition."""
    latvec = torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 4., 4.]])
    cutoff = torch.tensor([9.98])
    positions = torch.tensor([
        [0.949, 0.084, 0.020], [2.469, 0.084, 0.020], [0.573, 1.098, 0.268],
        [0.573, -0.638, 0.775], [0.573, -0.209, -0.982], [2.845, 0.376, 1.023],
        [2.845, 0.805, -0.735], [2.845, -0.931, -0.227]])
    numbers = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1])
    molecule = System(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    periodic = Periodic(molecule, molecule.cell, cutoff=cutoff)
    skt = SKT(molecule, sktable, periodic)
    coulomb = Coulomb(molecule, periodic)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter, coulomb, periodic)
    assert torch.max(abs(scc.charge - torch.tensor([
        4.1973198724049929, 4.1972128614775004, 0.93784050058296253,
        0.93230148676159263, 0.93253138404197577, 0.93268865020401870,
        0.93203620075479765, 0.93806904377216838]))) < 1E-8, 'Tolerance check'


def test_batch_pe():
    """Test scc batch calculation."""
    latvec = [torch.tensor([[4., 0., 0.], [0., 5., 0.], [0., 0., 4.]]),
              torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]])]
    cutoff = torch.tensor([9.98])
    positions = [torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]]),
         torch.tensor([[0., 0., 0.], [0., 2., 0.]])]
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([1, 1])]
    molecule = System(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    periodic = Periodic(molecule, molecule.cell, cutoff=cutoff)
    skt = SKT(molecule, sktable, periodic)
    coulomb = Coulomb(molecule, periodic)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter, coulomb, periodic)
    assert torch.max(abs(scc.charge - torch.tensor([[
        4.6319123620719083, 0.82123077572651604, 0.85069856480810935,
        0.84545973258536744, 0.85069856480810924],
        [1.000000000000000, 1.000000000000000, 0.000000000000000,
         0.000000000000000, 0.000000000000000]]))) < 1E-8, 'Tolerance check'


def test_batch_pe_2():
    """Test scc batch calculation."""
    latvec = [torch.tensor([[4., 4., 0.], [5., 0., 5.], [0., 6., 6.]]),
              torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]]),
              torch.tensor([[5., 0., 0.], [0., 5., 0.], [0., 0., 5.]]),
              torch.tensor([[99., 0., 0.], [0., 99., 0.], [0., 0., 99.]])]
    cutoff = torch.tensor([9.98])
    positions = [torch.tensor([
        [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]]),
         torch.tensor([[0., 0., 0.], [0., 2., 0.]]),
         torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056], [2.244, 0.660, 0.778]]),
         torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056], [2.244, 0.660, 0.778]])]
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([1, 1]),
               torch.tensor([1, 8, 1]), torch.tensor([1, 8, 1])]
    molecule = System(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    periodic = Periodic(molecule, molecule.cell, cutoff=cutoff)
    skt = SKT(molecule, sktable, periodic)
    coulomb = Coulomb(molecule, periodic)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter, coulomb, periodic)
    assert torch.max(abs(scc.charge - torch.tensor([[
            4.6122976812946259, 0.83320615382097674, 0.85273810385371818,
            0.85182728982744738, 0.84993077120323290],
            [1.000000000000000, 1.000000000000000, 0.000000000000000,
             0.000000000000000, 0.000000000000000],
            [0.70282850018606047, 6.5936446382800851, 0.70352686153385458,
             0.000000000000000, 0.000000000000000],
            [0.70794447853157250, 6.5839848726758881, 0.70807064879254611,
             0.000000000000000, 0.000000000000000]]))) < 1E-8, 'Tolerance check'


def test_batch_pe_npe():
    """Test scc batch calculation for mix of periodic and non-periodic systems."""
    latvec = [torch.tensor([[4., 4., 0.], [5., 0., 5.], [0., 6., 6.]]),
              torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
              torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]]),
              torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])]
    cutoff = torch.tensor([9.98])
    positions = [torch.tensor([
            [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]]),
        torch.tensor([
            [3., 3., 3.], [3.6, 3.6, 3.6], [2.4, 3.6, 3.6], [3.6, 2.4, 3.6], [3.6, 3.6, 2.4]]),
        torch.tensor([[0., 0., 0.], [0., 2., 0.]]),
        torch.tensor([[0.965, 0.075, 0.088], [1.954, 0.047, 0.056], [2.244, 0.660, 0.778]])]
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([6, 1, 1, 1, 1]),
               torch.tensor([1, 1]), torch.tensor([1, 8, 1])]
    molecule = System(numbers, positions, latvec)
    sktable = IntegralGenerator.from_dir('./slko/mio-1-1', molecule)
    periodic = Periodic(molecule, molecule.cell, cutoff=cutoff)
    skt = SKT(molecule, sktable, periodic)
    coulomb = Coulomb(molecule, periodic)
    parameter = Parameter()
    scc = Scc(molecule, skt, parameter, coulomb, periodic)
    assert torch.max(abs(scc.charge - torch.tensor([[
            4.6122976812946259, 0.83320615382097674, 0.85273810385371818,
            0.85182728982744738, 0.84993077120323290],
            [4.6010835947475499, 0.84036067839669026, 0.85285190895192031,
             0.85285190895192053, 0.85285190895192087],
            [1.000000000000000, 1.000000000000000, 0.000000000000000,
             0.000000000000000, 0.000000000000000],
            [0.70794502349564015, 6.5839837819000406, 0.70807119460432610,
             0.000000000000000, 0.000000000000000]]))) < 1E-8, 'Tolerance check'
