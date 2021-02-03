"""Perform tests on Basis in the tbmalt.common.structures.basis."""
import torch
from ase.build import molecule as molecule_database
from tbmalt.common.batch import pack
from tbmalt.common.structures.basis import Basis
from tbmalt.common.structures.system import System
torch.set_default_dtype(torch.float64)
torch.set_printoptions(15)


def test_basis_blocks_single(device):
    """Test single basis blocks and sub_blocks."""
    numbers = [torch.tensor([6, 1, 1, 1, 1])]
    sys = System(numbers, positions_ch4)
    basis = Basis(sys)
    blocks = basis._blocks()
    sub_blocks = basis._sub_blocks()

    # make sure each orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_block[0], blocks[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_sub_block[0], sub_blocks[0])]


def test_basis_blocks_batch(device):
    """Test batch basis blocks and sub_blocks."""
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([7, 1, 1, 1])]
    sys = System(numbers, pack([positions_ch4, positions_nh3]))
    basis = Basis(sys)
    blocks = basis._blocks()
    sub_blocks = basis._sub_blocks()

    # make sure each orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_block, blocks[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_sub_block, sub_blocks[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_nh3_block, blocks[1])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_nh3_sub_block, sub_blocks[1])]


def test_basis_sub_shells_single(device):
    """Test single system total sub shell number."""
    numbers = [torch.tensor([6, 1, 1, 1, 1])]
    sys = System(numbers, positions_ch4)
    basis = Basis(sys)
    sub_shells = basis._sub_shells()
    assert sub_shells[0] == 2 + 1 + 1 + 1 + 1


def test_basis_sub_shells_batch(device):
    """Test batch system total sub shell numbers."""
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([7, 1, 1, 1])]
    sys = System(numbers, pack([positions_ch4, positions_nh3]))
    basis = Basis(sys)
    sub_shells = basis._sub_shells()
    assert sub_shells[0] == 2 + 1 + 1 + 1 + 1, sub_shells[1] == 2 + 1 + 1 + 1


def test_basis_sub_basis_list_single(device):
    """Test single sub_basis_list."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    sys = System(numbers, positions_ch4)
    basis = Basis(sys)
    sub_shells = basis._sub_basis_list

    # make sure each sub orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(sub_basis_list_ch4, sub_shells[0])]


def test_basis_sub_basis_list_batch(device):
    """Test batch sub_basis_list."""
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([7, 1, 1, 1])]
    sys = System(numbers, pack([positions_ch4, positions_nh3]))
    basis = Basis(sys)
    sub_shells = basis._sub_basis_list

    # make sure each sub orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(sub_basis_list_ch4, sub_shells[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(sub_basis_list_nh3, sub_shells[1])]


def test_basis_azimuthal_matrix_single(device):
    """Test single azimuthal_matrix."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    sys = System(numbers, positions_ch4)
    basis = Basis(sys)
    mt_bt = basis.azimuthal_matrix(mask=True, block=True)
    mt_bf = basis.azimuthal_matrix(mask=True, block=False)
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_block, mt_bt[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_full, mt_bf[0])]


def test_basis_azimuthal_matrix_batch(device):
    """Test batch azimuthal_matrix."""
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([7, 1, 1, 1])]
    positions = pack([positions_ch4, positions_nh3])
    sys = System(numbers, positions)
    basis = Basis(sys)
    mt_bt = basis.azimuthal_matrix(mask=True, block=True)
    mt_bf = basis.azimuthal_matrix(mask=True, block=False)
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_block, mt_bt[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_full, mt_bf[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_nh3_l_block, mt_bt[1, :5, :5])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_nh3_l_full, mt_bf[1, :7, :7])]


def test_basis_atomic_number_single(device):
    """Test single atomic number."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    sys = System(numbers, positions_ch4)
    basis = Basis(sys)
    atom_num_atom = basis.atomic_number_matrix('atom')
    atom_num_block = basis.atomic_number_matrix('block')
    atom_num_orb = basis.atomic_number_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_atom_ch4, atom_num_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_block_ch4, atom_num_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_orbital_ch4, atom_num_orb[0])]


def test_basis_atomic_number_batch(device):
    """Test batch atomic number."""
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([7, 1, 1, 1])]
    positions = pack([positions_ch4, positions_nh3])
    sys = System(numbers, positions)
    basis = Basis(sys)
    atom_num_atom = basis.atomic_number_matrix('atom')
    atom_num_block = basis.atomic_number_matrix('block')
    atom_num_orb = basis.atomic_number_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_atom_ch4, atom_num_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_block_ch4, atom_num_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_orbital_ch4, atom_num_orb[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_atom_nh3, atom_num_atom[1, :4, :4])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_block_nh3, atom_num_block[1, :5, :5])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_orbital_nh3, atom_num_orb[1, :7, :7])]


def test_basis_index_matrix_single(device):
    """Test single index of atoms."""
    numbers = torch.tensor([6, 1, 1, 1, 1])
    sys = System(numbers, positions_ch4)
    basis = Basis(sys)
    index_atom = basis.index_matrix('atom')
    index_block = basis.index_matrix('block')
    index_orb = basis.index_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_atom_ch4, index_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_block_ch4, index_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_orbital_ch4, index_orb[0])]


def test_basis_index_matrix_batch(device):
    """Test batch index of atoms."""
    numbers = [torch.tensor([6, 1, 1, 1, 1]), torch.tensor([7, 1, 1, 1])]
    positions = pack([positions_ch4, positions_nh3])
    sys = System(numbers, positions)
    basis = Basis(sys)
    index_atom = basis.index_matrix('atom')
    index_block = basis.index_matrix('block')
    index_orb = basis.index_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_atom_ch4, index_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_block_ch4, index_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_orbital_ch4, index_orb[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_atom_nh3, index_atom[1, :4, :4])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_block_nh3, index_block[1, :5, :5])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_orbital_nh3, index_orb[1, :7, :7])]


def test_basis_blocks_single_ase(device):
    """Test single basis blocks and sub_blocks."""
    sys = System.from_ase_atoms([molecule_database('CH4')])
    basis = Basis(sys)
    blocks = basis._blocks()
    sub_blocks = basis._sub_blocks()

    # make sure each orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_block, blocks[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_sub_block, sub_blocks[0])]


def test_basis_blocks_batch_ase(device):
    """Test batch basis blocks and sub_blocks."""
    sys = System.from_ase_atoms([molecule_database('CH4'),
                                 molecule_database('NH3')])
    basis = Basis(sys)
    blocks = basis._blocks()
    sub_blocks = basis._sub_blocks()

    # make sure each orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_block, blocks[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_ch4_sub_block, sub_blocks[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_nh3_block, blocks[1])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(ref_nh3_sub_block, sub_blocks[1])]


def test_basis_sub_shells_single_ase(device):
    """Test single system total sub shell numbers."""
    sys = System.from_ase_atoms([molecule_database('CH4')])
    basis = Basis(sys)
    sub_shells = basis._sub_shells()
    assert sub_shells[0] == 2 + 1 + 1 + 1 + 1


def test_basis_sub_shells_batch_ase(device):
    """Test batch system total sub shell numbers."""
    sys = System.from_ase_atoms([molecule_database('CH4'),
                                 molecule_database('NH3')])
    basis = Basis(sys)
    sub_shells = basis._sub_shells()
    assert sub_shells[0] == 2 + 1 + 1 + 1 + 1, sub_shells[1] == 2 + 1 + 1 + 1


def test_basis_sub_basis_list_single_ase(device):
    """Test single sub_basis_list."""
    sys = System.from_ase_atoms([molecule_database('CH4')])
    basis = Basis(sys)
    sub_shells = basis._sub_basis_list

    # make sure each sub orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(sub_basis_list_ch4, sub_shells[0])]


def test_basis_sub_basis_list_batch_ase(device):
    """Test batch sub_basis_list."""
    sys = System.from_ase_atoms([molecule_database('CH4'),
                                 molecule_database('NH3')])
    basis = Basis(sys)
    sub_shells = basis._sub_basis_list

    # make sure each sub orbital blocks is the same as reference
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(sub_basis_list_ch4, sub_shells[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(sub_basis_list_nh3, sub_shells[1])]


def test_basis_azimuthal_matrix_single_ase(device):
    """Test single azimuthal_matrix."""
    sys = System.from_ase_atoms([molecule_database('CH4')])
    basis = Basis(sys)
    mt_bt = basis.azimuthal_matrix(mask=True, block=True)
    mt_bf = basis.azimuthal_matrix(mask=True, block=False)
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_block, mt_bt[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_full, mt_bf[0])]


def test_basis_azimuthal_matrix_batch_ase(device):
    """Test batch azimuthal_matrix."""
    sys = System.from_ase_atoms([molecule_database('CH4'),
                                 molecule_database('NH3')])
    basis = Basis(sys)
    mt_bt = basis.azimuthal_matrix(mask=True, block=True)
    mt_bf = basis.azimuthal_matrix(mask=True, block=False)
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_block, mt_bt[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_ch4_l_full, mt_bf[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_nh3_l_block, mt_bt[1, :5, :5])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(azimuthal_nh3_l_full, mt_bf[1, :7, :7])]


def test_basis_atomic_number_single_ase(device):
    """Test single atomic number."""
    sys = System.from_ase_atoms([molecule_database('CH4')])
    basis = Basis(sys)
    atom_num_atom = basis.atomic_number_matrix('atom')
    atom_num_block = basis.atomic_number_matrix('block')
    atom_num_orb = basis.atomic_number_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_atom_ch4, atom_num_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_block_ch4, atom_num_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_orbital_ch4, atom_num_orb[0])]


def test_basis_atomic_number_batch_ase(device):
    """Test batch atomic number."""
    sys = System.from_ase_atoms([molecule_database('CH4'),
                                 molecule_database('NH3')])
    basis = Basis(sys)
    atom_num_atom = basis.atomic_number_matrix('atom')
    atom_num_block = basis.atomic_number_matrix('block')
    atom_num_orb = basis.atomic_number_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_atom_ch4, atom_num_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_block_ch4, atom_num_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_orbital_ch4, atom_num_orb[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_atom_nh3, atom_num_atom[1, :4, :4])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_block_nh3, atom_num_block[1, :5, :5])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(atomnumber_orbital_nh3, atom_num_orb[1, :7, :7])]


def test_basis_index_matrix_single_ase(device):
    """Test single index of atoms."""
    sys = System.from_ase_atoms([molecule_database('CH4')])
    basis = Basis(sys)
    index_atom = basis.index_matrix('atom')
    index_block = basis.index_matrix('block')
    index_orb = basis.index_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_atom_ch4, index_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_block_ch4, index_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_orbital_ch4, index_orb[0])]


def test_basis_index_matrix_batch_ase(device):
    """Test batch index of atoms."""
    sys = System.from_ase_atoms([molecule_database('CH4'),
                                 molecule_database('NH3')])
    basis = Basis(sys)
    index_atom = basis.index_matrix('atom')
    index_block = basis.index_matrix('block')
    index_orb = basis.index_matrix('orbital')
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_atom_ch4, index_atom[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_block_ch4, index_block[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_orbital_ch4, index_orb[0])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_atom_nh3, index_atom[1, :4, :4])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_block_nh3, index_block[1, :5, :5])]
    assert False not in [(ii == jj).all() for ii, jj in
                         zip(index_orbital_nh3, index_orb[1, :7, :7])]


positions_ch4 = torch.tensor([[0.0000000000, 0.0000000000, 0.0000000000],
                              [0.6287614522, 0.6287614522, 0.6287614522],
                              [-0.6287614522, -0.6287614522, 0.6287614522],
                              [-0.6287614522, 0.6287614522, -0.6287614522],
                              [0.6287614522, -0.6287614522, -0.6287614522]])

positions_nh3 = torch.tensor([
    [0., 0., 0.116489], [0., 0.939731, -0.271808],
    [0.813831, -0.469865, -0.271808], [-0.813831, -0.469865, -0.271808]])

ref_ch4_block = [torch.full((4, 4), True, dtype=torch.bool),
                 torch.full((1, 1), True, dtype=torch.bool),
                 torch.full((1, 1), True, dtype=torch.bool),
                 torch.full((1, 1), True, dtype=torch.bool),
                 torch.full((1, 1), True, dtype=torch.bool)]

ref_ch4_sub_block = [torch.full((2, 2), True, dtype=torch.bool),
                     torch.full((1, 1), True, dtype=torch.bool),
                     torch.full((1, 1), True, dtype=torch.bool),
                     torch.full((1, 1), True, dtype=torch.bool),
                     torch.full((1, 1), True, dtype=torch.bool)]

ref_nh3_block = [torch.full((4, 4), True, dtype=torch.bool),
                 torch.full((1, 1), True, dtype=torch.bool),
                 torch.full((1, 1), True, dtype=torch.bool),
                 torch.full((1, 1), True, dtype=torch.bool)]

ref_nh3_sub_block = [torch.full((2, 2), True, dtype=torch.bool),
                     torch.full((1, 1), True, dtype=torch.bool),
                     torch.full((1, 1), True, dtype=torch.bool),
                     torch.full((1, 1), True, dtype=torch.bool)]

sub_basis_list_ch4 = [torch.tensor([0, 1]), torch.tensor([0]),
                      torch.tensor([0]), torch.tensor([0]), torch.tensor([0])]

sub_basis_list_nh3 = [torch.tensor([0, 1]), torch.tensor([0]),
                      torch.tensor([0]), torch.tensor([0])]

azimuthal_ch4_l_full = torch.tensor([
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[1, 0], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[1, 0], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[1, 0], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]])

# onsiteis -1, the others will l pairs
azimuthal_ch4_l_block = torch.tensor([
    [[-1, -1], [-1, -1], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[0, 0], [0, 1], [-1, -1], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 0], [-1, -1], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 0], [0, 0], [-1, -1], [0, 0]],
    [[0, 0], [0, 1], [0, 0], [0, 0], [0, 0], [-1, -1]]])

azimuthal_ch4_l_full = torch.tensor([
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0], [1, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [-1, -1], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [-1, -1], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [-1, -1], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0], [-1, -1]]])

azimuthal_nh3_l_block = torch.tensor([
    [[-1, -1], [-1, -1], [0, 0], [0, 0], [0, 0]],
    [[-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0]],
    [[0, 0], [0, 1], [-1, -1], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 0], [-1, -1], [0, 0]],
    [[0, 0], [0, 1], [0, 0], [0, 0], [-1, -1]]])

azimuthal_nh3_l_full = torch.tensor([
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [0, 0], [0, 0], [0, 0]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0]],
    [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [1, 0], [1, 0], [1, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [-1, -1], [0, 0], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [-1, -1], [0, 0]],
    [[0, 0], [0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [-1, -1]]])

atomnumber_atom_ch4 = torch.tensor([
    [[6, 6], [6, 1], [6, 1], [6, 1], [6, 1]],
    [[1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 1], [1, 1], [1, 1], [1, 1]]])

atomnumber_block_ch4 = torch.tensor([
    [[6, 6], [6, 6], [6, 1], [6, 1], [6, 1], [6, 1]],
    [[6, 6], [6, 6], [6, 1], [6, 1], [6, 1], [6, 1]],
    [[1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]]])

atomnumber_orbital_ch4 = torch.tensor([
    [[6, 6], [6, 6], [6, 6], [6, 6], [6, 1], [6, 1], [6, 1], [6, 1]],
    [[6, 6], [6, 6], [6, 6], [6, 6], [6, 1], [6, 1], [6, 1], [6, 1]],
    [[6, 6], [6, 6], [6, 6], [6, 6], [6, 1], [6, 1], [6, 1], [6, 1]],
    [[6, 6], [6, 6], [6, 6], [6, 6], [6, 1], [6, 1], [6, 1], [6, 1]],
    [[1, 6], [1, 6], [1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 6], [1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 6], [1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]],
    [[1, 6], [1, 6], [1, 6], [1, 6], [1, 1], [1, 1], [1, 1], [1, 1]]])

atomnumber_atom_nh3 = torch.tensor([
    [[7, 7], [7, 1], [7, 1], [7, 1]],
    [[1, 7], [1, 1], [1, 1], [1, 1]],
    [[1, 7], [1, 1], [1, 1], [1, 1]],
    [[1, 7], [1, 1], [1, 1], [1, 1]]])

atomnumber_block_nh3 = torch.tensor([
    [[7, 7], [7, 7], [7, 1], [7, 1], [7, 1]],
    [[7, 7], [7, 7], [7, 1], [7, 1], [7, 1]],
    [[1, 7], [1, 7], [1, 1], [1, 1], [1, 1]],
    [[1, 7], [1, 7], [1, 1], [1, 1], [1, 1]],
    [[1, 7], [1, 7], [1, 1], [1, 1], [1, 1]]])

atomnumber_orbital_nh3 = torch.tensor([
    [[7, 7], [7, 7], [7, 7], [7, 7], [7, 1], [7, 1], [7, 1]],
    [[7, 7], [7, 7], [7, 7], [7, 7], [7, 1], [7, 1], [7, 1]],
    [[7, 7], [7, 7], [7, 7], [7, 7], [7, 1], [7, 1], [7, 1]],
    [[7, 7], [7, 7], [7, 7], [7, 7], [7, 1], [7, 1], [7, 1]],
    [[1, 7], [1, 7], [1, 7], [1, 7], [1, 1], [1, 1], [1, 1]],
    [[1, 7], [1, 7], [1, 7], [1, 7], [1, 1], [1, 1], [1, 1]],
    [[1, 7], [1, 7], [1, 7], [1, 7], [1, 1], [1, 1], [1, 1]]])

index_atom_ch4 = torch.tensor([
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
    [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
    [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
    [[3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
    [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]])

index_block_ch4 = torch.tensor([
    [[0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
    [[0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
    [[1, 0], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
    [[2, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
    [[3, 0], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
    [[4, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]])

index_orbital_ch4 = torch.tensor([
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
    [[2, 0], [2, 0], [2, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]],
    [[3, 0], [3, 0], [3, 0], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4]],
    [[4, 0], [4, 0], [4, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]])

index_atom_nh3 = torch.tensor([
    [[0, 0], [0, 1], [0, 2], [0, 3]],
    [[1, 0], [1, 1], [1, 2], [1, 3]],
    [[2, 0], [2, 1], [2, 2], [2, 3]],
    [[3, 0], [3, 1], [3, 2], [3, 3]]])

index_block_nh3 = torch.tensor([
    [[0, 0], [0, 0], [0, 1], [0, 2], [0, 3]],
    [[0, 0], [0, 0], [0, 1], [0, 2], [0, 3]],
    [[1, 0], [1, 0], [1, 1], [1, 2], [1, 3]],
    [[2, 0], [2, 0], [2, 1], [2, 2], [2, 3]],
    [[3, 0], [3, 0], [3, 1], [3, 2], [3, 3]]])

index_orbital_nh3 = torch.tensor([
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3]],
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 3]],
    [[1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 2], [1, 3]],
    [[2, 0], [2, 0], [2, 0], [2, 0], [2, 1], [2, 2], [2, 3]],
    [[3, 0], [3, 0], [3, 0], [3, 0], [3, 1], [3, 2], [3, 3]]])
