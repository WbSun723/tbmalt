"""Performs tests on functions present in the tbmalt.common.maths module"""
import torch
from ase.atoms import Atoms
import h5py
from tbmalt.common.structures.system import System
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)
_bohr = 0.529177249


def _calculate_distance(numbers, positions):
    """The traditional way to test the distance calculation."""
    positions = positions.unsqueeze(0) if positions.dim() == 2 else positions
    numbers = numbers.unsqueeze(0) if numbers.dim() == 1 else numbers
    natom = [len(inum) for inum in numbers]
    distance = torch.zeros(len(positions), max(natom), max(natom))
    for ib, iat in enumerate(natom):
        for ii, ipos in enumerate(positions[ib, :iat]):
            for jj, jpos in enumerate(positions[ib, :iat]):
                distance[ib, ii, jj] = torch.sqrt(sum((ipos - jpos) ** 2))
    return distance


def test_system_single(device):
    """Test single system input."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    positions = torch.tensor([[0.0000000000, 0.0000000000, 0.0000000000],
                              [0.6287614522, 0.6287614522, 0.6287614522],
                              [-0.6287614522, -0.6287614522, 0.6287614522],
                              [-0.6287614522, 0.6287614522, -0.6287614522],
                              [0.6287614522, -0.6287614522, -0.6287614522]])
    sys = System(numbers, positions)
    distance_ref = _calculate_distance(numbers, positions / _bohr)
    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['C', 'H', 'H', 'H', 'H']]
    assert sys.size_batch == 1
    assert sys.size_system == [5]
    assert sys.l_max == [[1, 0, 0, 0, 0]]
    assert sys.hs_shape == torch.Size([1, 8, 8])


def test_symtem_ase_byhand_single(device):
    """Test single input, deal with ase input by hand."""
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])

    # deal with input by hand
    positions = torch.from_numpy(ch4.positions)
    numbers = torch.from_numpy(ch4.numbers)

    sys = System(numbers, positions)

    # reference
    distance_ref = _calculate_distance(numbers, positions / _bohr)

    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['C', 'H', 'H', 'H', 'H']]
    assert sys.size_batch == 1
    assert sys.size_system == [5]
    assert sys.l_max == [[1, 0, 0, 0, 0]]
    assert sys.hs_shape == torch.Size([1, 8, 8])


def test_system_ase_byhand_batch(device):
    """Batch evaluation of maths.sym function."""
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    h2 = Atoms('H2', positions=[[0., 0., 0.], [0.5, 0.5, 0.5]])

    # deal with input by hand
    positions = [torch.from_numpy(ch4.positions), torch.from_numpy(h2.positions)]
    numbers = [torch.from_numpy(ch4.numbers), torch.from_numpy(h2.numbers)]

    sys = System(numbers, positions)

    distance_ref = [_calculate_distance(num, pos / _bohr)
                    for num, pos in zip(numbers, positions)]

    assert torch.max(abs(sys.distances[0] - distance_ref[0])) < 1E-14
    assert torch.max(abs(sys.distances[1, :2, :2] - distance_ref[1])) < 1E-14
    assert sys.symbols[0] == ['C', 'H', 'H', 'H', 'H']
    assert sys.symbols[1] == ['H', 'H']
    assert sys.size_batch == 2
    assert sys.size_system == [5, 2]
    assert sys.l_max == [[1, 0, 0, 0, 0], [0, 0]]
    assert sys.hs_shape == torch.Size([2, 8, 8])


def test_from_ase_single(device):
    """Batch evaluation of maths.sym function."""
    # ase Atoms as input for single system
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    sys = System.from_ase_atoms(ch4)

    positions = torch.from_numpy(ch4.positions)
    numbers = torch.from_numpy(ch4.numbers)
    distance_ref = _calculate_distance(numbers, positions / _bohr)

    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert torch.max(abs(sys.distances - distance_ref)) < 1E-14
    assert sys.symbols == [['C', 'H', 'H', 'H', 'H']]
    assert sys.size_batch == 1
    assert sys.size_system == [5]
    assert sys.l_max == [[1, 0, 0, 0, 0]]
    assert sys.hs_shape == torch.Size([1, 8, 8])


def test_from_ase_batch(device):
    """Batch evaluation of maths.sym function."""
    # ase Atoms as input for batch system
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    h2 = Atoms('H2', positions=[[0., 0., 0.], [0.5, 0.5, 0.5]])
    sys = System.from_ase_atoms([ch4, h2])

    positions = [torch.from_numpy(ch4.positions), torch.from_numpy(h2.positions)]
    numbers = [torch.from_numpy(ch4.numbers), torch.from_numpy(h2.numbers)]
    distance_ref = [_calculate_distance(num, pos / _bohr)
                    for num, pos in zip(numbers, positions)]

    assert torch.max(abs(sys.distances[0] - distance_ref[0])) < 1E-14
    assert torch.max(abs(sys.distances[1, :2, :2] - distance_ref[1])) < 1E-14
    assert sys.symbols[0] == ['C', 'H', 'H', 'H', 'H']
    assert sys.symbols[1] == ['H', 'H']
    assert sys.size_batch == 2
    assert sys.size_system == [5, 2]
    assert sys.l_max == [[1, 0, 0, 0, 0], [0, 0]]
    assert sys.hs_shape == torch.Size([2, 8, 8])


def test_hdf5():
    """Test h5py."""
    ch4 = Atoms('CH4', positions=[
        [0., 0., 0.], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])
    positions = torch.from_numpy(ch4.positions)
    numbers = torch.from_numpy(ch4.numbers)
    with h5py.File('test.hdf5', 'w') as f:
        System(numbers, positions).to_hd5(f)
