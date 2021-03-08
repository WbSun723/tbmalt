"""Test alpha and 1/R matrix calculation for coulomb interaction in periodic system."""

import torch
import numpy as np
from tbmalt.common.batch import pack
from tbmalt.common.structures.geometry import Geometry
from tbmalt.tb.coulomb import Coulomb
from tbmalt.common.structures.periodic import Periodic
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
_bohr = 0.529177249


def test_get_alpha():
    """Test optimizing alpha for the Ewald sum."""
    latvec = torch.tensor([[0, 4., 0], [2., 0, 0], [0, 0, 2.]])
    coord = torch.tensor([[0, 0, 0], [0, 2., 0]])
    atompair = torch.tensor([1, 1])
    geo = Geometry(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.alpha - alpha_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_alpha_batch():
    """Test optimizing alphas with different lattice vectors for the Ewald sum."""
    latvec = [torch.tensor([[0, 4, 0], [2, 0, 0], [0, 0, 2]]),
              torch.tensor([[0, 8, 0], [4, 0, 0], [0, 0, 4]])]
    positions = torch.tensor([[[0, 0, 0], [0, 2, 0]], [[0, 0, 0], [0, 2, 0]]]) / _bohr
    atompair = [torch.tensor([1, 1]), torch.tensor([1, 1])]
    geo = Geometry(atompair, positions, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.alpha - alpha_batch_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_h2():
    """Test 1/R matrix calculation in periodic system."""
    latvec = torch.tensor([[0, 4., 0], [2., 0, 0], [0, 0, 2.]])
    coord = torch.tensor([[0, 0, 0], [0, 2., 0]])
    atompair = torch.tensor([1, 1])
    geo = Geometry(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat[0] - invr_h2_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch():
    """Test 1/R matrix batch calculation in periodic system."""
    latvec = torch.tensor([[0, 8, 0], [4, 0, 0], [0, 0, 4]]).repeat(3, 1, 1)

    # Batch calculation for h2, ch4 and co2 periodic systems
    atompair = [torch.tensor([1, 1]), torch.tensor([6, 1, 1, 1, 1]), torch.tensor([8, 6, 8])]
    positions = [torch.tensor([[0, 0, 0], [0, 4., 0]]),
                 torch.tensor([[0, 0, 0], [0.6287614522, 0.6287614522, 0.6287614522],
                               [-0.6287614522, -0.6287614522, 0.6287614522],
                               [-0.6287614522, 0.6287614522, -0.6287614522],
                               [0.6287614522, -0.6287614522, -0.6287614522]]),
                 torch.tensor([[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
                               [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
                               [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]])]
    geo = Geometry(atompair, positions, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat - invr_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch_difflatvec():
    """Test 1/R matrix batch calculation with different lattice vectors as input in periodic system."""
    latvec = [torch.tensor([[0, 4., 0], [2., 0, 0], [0, 0, 2.]]),
              torch.tensor([[0, 8., 0], [4., 0, 0], [0, 0, 4.]])]

    atompair = [torch.tensor([1, 1]), torch.tensor([8, 6, 8])]
    coord = [torch.tensor([[0, 0, 0], [0, 2., 0]]),
             torch.tensor([[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
                           [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
                           [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]])]
    geo = Geometry(atompair, coord, latvec)
    periodic = Periodic(geo, geo.cell, cutoff=10.)
    coulomb = Coulomb(geo, periodic)
    assert torch.max(abs(coulomb.invrmat - invr2_from_dftbplus)) < 1E-14, 'Tolerance check'


alpha_from_dftbplus = torch.tensor([0.47513600000000000])
alpha_batch_from_dftbplus = torch.tensor([0.47513600000000000, 0.23756800000000000])
invr_h2_from_dftbplus = torch.tensor([[-0.47780521676190824, -0.27291142548538699],
                                      [-0.27291142548538699, -0.47780521676190824]])
invr2_h2_from_dftbplus = torch.tensor([[-0.23890261448525388, -0.13645572591320607],
                                       [-0.13645572591320607, -0.23890261448525388]])
invr_ch4_from_dftbplus = torch.tensor([[-0.23890261448525388, 0.25618239418944322, 0.25618239418944333, 0.25618239418944333, 0.25618239418944333],
                                       [0.25618239418944322, -0.23890261448525388, 6.7967077421936944E-002, 0.11166535647645823, 6.7967077421936944E-002],
                                       [0.25618239418944322, 6.7967077421936944E-002, -0.23890261448525388, 6.7967077421936889E-002, 0.11166535647645823],
                                       [0.25618239418944322, 0.11166535647645823, 6.7967077421936944E-002, -0.23890261448525388, 6.7967077421936944E-002],
                                       [0.25618239418944322, 6.7967077421936944E-002, 0.11166535647645823, 6.7967077421936944E-002, -0.23890261448525388]])
invr_co2_from_dftbplus = torch.tensor([[-0.23890261448525388, 0.25142867181718609, 0.13878449166434717],
                                       [0.25142867181718609, -0.23890261448525388, 0.24402908422714895],
                                       [0.13878449166434717, 0.24402908422714895, -0.23890261448525388]])
invr_from_dftbplus = pack([invr2_h2_from_dftbplus, invr_ch4_from_dftbplus, invr_co2_from_dftbplus])
invr2_from_dftbplus = pack([invr_h2_from_dftbplus, invr_co2_from_dftbplus])
