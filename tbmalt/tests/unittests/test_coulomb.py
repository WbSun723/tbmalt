"""Test alpha and 1/R matrix calculation for coulomb interaction in periodic system."""

import torch
import numpy as np
from tbmalt.common.batch import pack
from tbmalt.common.structures.system import System
from tbmalt.tb.coulomb import Coulomb
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
_bohr = 0.529177249


def test_get_alpha():
    """Test optimizing alpha for the Ewald sum."""
    latvec = torch.tensor([[0, 4, 0], [2, 0, 0], [0, 0, 2]]) / _bohr
    recvec = 2.0 * np.pi * torch.transpose(torch.inverse(latvec), 0, 1)
    coord = torch.tensor([[0, 0, 0], [0, 2, 0]]) / _bohr
    atompair = torch.tensor([1, 1])
    system = System(atompair, coord, latvec)
    coulomb = Coulomb(system, latvec, recvec)
    assert torch.max(abs(coulomb.alpha - alpha_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_alpha_batch():
    """Test optimizing alphas with different lattice vectors for the Ewald sum."""
    latvec1 = torch.tensor([[0, 4, 0], [2, 0, 0], [0, 0, 2]]) / _bohr
    latvec2 = torch.tensor([[0, 8, 0], [4, 0, 0], [0, 0, 4]]) / _bohr
    recvec1 = 2.0 * np.pi * torch.transpose(torch.inverse(latvec1), 0, 1)
    recvec2 = 2.0 * np.pi * torch.transpose(torch.inverse(latvec2), 0, 1)
    latvec = torch.cat((torch.unsqueeze(latvec1, 0), torch.unsqueeze(latvec2, 0)))
    recvec = torch.cat((torch.unsqueeze(recvec1, 0), torch.unsqueeze(recvec2, 0)))
    coord1 = torch.tensor([[0, 0, 0], [0, 2, 0]]) / _bohr
    atompair = [torch.tensor([1, 1]), torch.tensor([1, 1])]
    coord = pack([coord1, coord1])
    system = System(atompair, coord, latvec)
    coulomb = Coulomb(system, latvec, recvec)
    assert torch.max(abs(coulomb.alpha - alpha_batch_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_h2():
    """Test 1/R matrix calculation in periodic system."""
    latvec = torch.tensor([[0, 4, 0], [2, 0, 0], [0, 0, 2]]) / _bohr
    recvec = 2.0 * np.pi * torch.transpose(torch.inverse(latvec), 0, 1)
    coord = torch.tensor([[0, 0, 0], [0, 2, 0]]) / _bohr
    atompair = torch.tensor([1, 1])
    system = System(atompair, coord, latvec)
    coulomb = Coulomb(system, latvec, recvec)
    assert torch.max(abs(coulomb.invrmat[0] - invr_h2_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch():
    """Test 1/R matrix batch calculation in periodic system."""
    latvec1 = torch.tensor([[0, 8, 0], [4, 0, 0], [0, 0, 4]]) / _bohr
    recvec1 = 2.0 * np.pi * torch.transpose(torch.inverse(latvec1), 0, 1)
    latvec = torch.cat((torch.unsqueeze(latvec1, 0), torch.unsqueeze(latvec1, 0), torch.unsqueeze(latvec1, 0)))
    recvec = torch.cat((torch.unsqueeze(recvec1, 0), torch.unsqueeze(recvec1, 0), torch.unsqueeze(recvec1, 0)))

    # Batch calculation for h2, ch4 and co2 periodic systems
    atompair = [torch.tensor([1, 1]), torch.tensor([6, 1, 1, 1, 1]), torch.tensor([8, 6, 8])]
    coord1 = torch.tensor([[0, 0, 0], [0, 4, 0]]) / _bohr
    coord2 = torch.tensor([[0,0,0], [0.6287614522, 0.6287614522, 0.6287614522],
                       [-0.6287614522, -0.6287614522, 0.6287614522],
                       [-0.6287614522, 0.6287614522, -0.6287614522],
                       [0.6287614522, -0.6287614522, -0.6287614522]]) / _bohr
    coord3 = torch.tensor([[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
             [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
             [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]]) / _bohr
    coord = pack([coord1, coord2, coord3])
    system = System(atompair, coord, latvec)
    coulomb = Coulomb(system, latvec, recvec)
    assert torch.max(abs(coulomb.invrmat - invr_from_dftbplus)) < 1E-14, 'Tolerance check'


def test_get_invr_batch_difflatvec():
    """Test 1/R matrix batch calculation with different lattice vectors as input in periodic system."""
    latvec1 = torch.tensor([[0, 4, 0], [2, 0, 0], [0, 0, 2]]) / _bohr
    latvec2 = torch.tensor([[0, 8, 0], [4, 0, 0], [0, 0, 4]]) / _bohr
    recvec1 = 2.0 * np.pi * torch.transpose(torch.inverse(latvec1), 0, 1)
    recvec2 = 2.0 * np.pi * torch.transpose(torch.inverse(latvec2), 0, 1)
    latvec = torch.cat((torch.unsqueeze(latvec1, 0), torch.unsqueeze(latvec2, 0)))
    recvec = torch.cat((torch.unsqueeze(recvec1, 0), torch.unsqueeze(recvec2, 0)))

    # Batch calculation for h2 and co2 periodic systems with different lattice vectors
    atompair = [torch.tensor([1, 1]), torch.tensor([8, 6, 8])]
    coord1 = torch.tensor([[0, 0, 0], [0, 2, 0]]) / _bohr
    coord2 = torch.tensor([[-2.0357279573e-03, -1.7878314480e-02, 1.1467019320e+00],
             [5.4268823005e-03,  4.7660354525e-02, 7.7558560297e-03],
             [-2.0357279573e-03, -1.7878314480e-02, -1.1525206566e+00]]) / _bohr
    coord = pack([coord1, coord2])
    system = System(atompair, coord, latvec)
    coulomb = Coulomb(system, latvec, recvec)
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
