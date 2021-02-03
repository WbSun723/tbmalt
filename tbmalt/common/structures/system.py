"""Structural and orbital information."""
import torch
from typing import Union, List
from tbmalt.common.batch import pack
_bohr = 0.529177249
Tensor = torch.Tensor
_atom_name = ["H", "He",
              "Li", "Be", "B", "C", "N", "O", "F", "Ne",
              "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
              "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
              "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
              "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
              "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
              "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
              "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os",
              "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At"]

_val_elect = [1, 2,
              1, 2, 3, 4, 5, 6, 7, "Ne",
              1, 2, 3, 4, 5, 6, 7, "Ar",
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
              "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
              "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
              "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
              "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
              "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W ", "Re", "Os",
              "Ir", "Pt", 11, "Hg", "Tl", "Pb", "Bi", "Po", "At"]

_l_num = [0, 0,
          0, 0, 1, 1, 1, 1, 1, 1,
          0, 0, 1, 1, 1, 1, 1, 1,
          0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
          0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
          0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]


class System:
    r"""System object.

    This object will generate single system (molecule, unit cell) information,
    or a list of systems information from dataset. The difference of single and
    multi systems input will be the dimension.
    In general, the output information will include symbols, max quantum
    number, angular moments, magnetic moments, masses and atomic charges.

    Arguments:
        numbers: Atomic number of each atom in single or multi system.
            For multi systems, if system sizes is not same, then use sequences
            tensor input.
        positions: :math:`(N, 3)` where `N = number of atoms`
            in single system, or :math:`(M, N, 3)` where `M = number of
            systems, N = number of atoms` in multi systems.
        cell: optional
            The lattice vectors of the cell, if applicable. This should be
            a 3x3 matrix. While a 1x3 vector can be specified it will be
            auto parsed into a 3x3 vector.
        pbc: True will enact periodic boundary conditions (PBC) along all
            cell dimensions (buck systems), False will fully disable PBC
            (isolated molecules), & an array of booleans will only enact
            PBC on a subset of cell dimensions (slab). The first two can
            be auto-inferred from the presence of the ``cell`` parameter.
            (`bool`, `array_like` [`bool`], optional).

    Examples:
        >>> from ase.build import molecule as molecule_database
        >>> from tbmalt.common.structures.system import System
        >>> molecule = molecule_database('CH4')
        >>> system = System.from_ase_atoms(molecule)
        >>> system.numbers
        >>> tensor([[6, 1, 1, 1, 1]])
        >>> system = System(torch.tensor([1, 1]), torch.tensor([
            [0., 0., 0.], [0.5, 0.5, 0.5]]))
        >>> system.distances
        >>> tensor([[[0.0000, 1.6366], [1.6366, 0.0000]]])

    Todo:
        Add periodic boundaries.
    """

    def __init__(self, numbers: Union[Tensor, List[Tensor]],
                 positions: Union[Tensor, List[Tensor]],
                 lattice=None, **kwargs):
        self.pbc = kwargs['pbc'] if 'pbc' in kwargs else lattice is not None
        self.positions, self.numbers, self.batch = self._check(
            numbers, positions, **kwargs)

        # size of each system
        self.size_system = self._get_size()

        # get distance
        self.distances = self._get_distances()

        # get symbols
        self.symbols = self._get_symbols()

        # size of batch size, size of each system (number of atoms)
        self.size_batch = len(self.numbers)

        # get max l of each atom, number of orbitals (l + 1) ** 2 of each atom
        # number of total orbitals of each system in batch
        self.l_max, self.atom_orbitals, self.system_orbitals = \
            self._get_l_orbital()

        # get Hamiltonian, overlap shape in batch of each system and batch
        self.shape, self.hs_shape = self._get_hs_shape()

    def _check(self, numbers, positions, **kwargs):
        """Check the type and dimension of numbers, positions."""
        unit = kwargs['unit'] if 'unit' in kwargs else 'angstrom'

        # sequences of tensor
        if type(numbers) is list:
            numbers = pack(numbers)
        elif type(numbers) is torch.Tensor and numbers.dim() == 1:
            numbers = numbers.unsqueeze(0)

        # positions type check
        if type(positions) is list:
            positions = pack(positions)
        elif type(positions) is torch.Tensor and positions.dim() == 2:
            positions = positions.unsqueeze(0)

        # transfer positions from angstrom to bohr
        if unit in ('angstrom', 'Angstrom'):
            positions = positions / _bohr
        elif unit in ('bohr', 'Bohr'):
            positions = positions
        else:
            raise ValueError('Please select either angstrom or bohr')

        assert positions.shape[0] == numbers.shape[0]
        batch_ = True if numbers.shape[0] != 1 else False

        return positions, numbers, batch_

    def _get_distances(self):
        """Return distances between a list of atoms for each system."""
        return pack([torch.sqrt(((ipos[:inat].repeat(inat, 1) -
                                  ipos[:inat].repeat_interleave(inat, 0))
                                 ** 2).sum(1)).reshape(inat, inat)
                     for ipos, inat in zip(self.positions, self.size_system)])

    def _get_symbols(self):
        """Get atom name for each system in batch."""
        return [[_atom_name[ii - 1] for ii in inu[inu.ne(0.)]]
                for inu in self.numbers]

    def _get_size(self):
        """Get each system size (number of atoms) in batch."""
        return [len(inum[inum.ne(0.)]) for inum in self.numbers]

    def _get_l_orbital(self):
        """Return the number of orbitals associated with each atom."""
        # max l for each atom
        l_max = pack([torch.tensor([_l_num[ii - 1] for ii in inum[inum.ne(0)]])
                      for inum in self.numbers], value=-1)

        # max valence orbital number for each atom and each system
        atom_orbitals = pack([torch.tensor([(ii + 1) ** 2 for ii in lm])
                              for lm in l_max])
        system_orbitals = pack([sum(iao) for iao in atom_orbitals])

        return l_max, atom_orbitals, system_orbitals

    def _get_hs_shape(self):
        """Return shapes of Hamiltonian and overlap."""
        maxorb = max(self.system_orbitals)
        shape = [torch.Size([iorb, iorb]) for iorb in self.system_orbitals]
        hs_shape = torch.Size([self.size_batch, maxorb, maxorb])
        return shape, hs_shape

    def get_positions_vec(self):
        """Return positions vector between atoms."""
        return pack([ipo.unsqueeze(-2) - ipo.unsqueeze(-3)
                     for ipo in self.positions])

    def get_valence_electrons(self, dtype=torch.int64):
        """Return the number of orbitals associated with each atom."""
        # max l for each atom
        return pack([torch.tensor([_val_elect[ii - 1]
                                   for ii in inum[inum.ne(0)]], dtype=dtype)
                     for inum in self.numbers])

    def get_global_species(self):
        """Get global element species for single or multi systems."""
        numbers_ = torch.unique(self.numbers)
        numbers = numbers_[numbers_.ne(0.)]
        element_name = [_atom_name[ii - 1] for ii in numbers]
        element_number, nn = numbers.tolist(), len(numbers)
        element_name_pair = [[iel, jel] for iel, jel in zip(
            sorted(element_name * nn), element_name * nn)]
        element_number_pair = [
            [_atom_name.index(ii[0]) + 1, _atom_name.index(ii[1]) + 1]
            for ii in element_name_pair]
        return element_name, element_number, element_name_pair, element_number_pair

    def get_resolved_orbital(self):
        """Return resolved orbitals and accumulated orbitals."""
        return pack([torch.cat([torch.arange(
            lm + 1, dtype=torch.int8).repeat_interleave(2 * torch.arange(
                lm + 1) + 1) for lm in ilm]) for ilm in self.l_max], value=-1)

    @classmethod
    def to_element_number(cls, element: list):
        """Return element number from element."""
        element = [element] if type(element[0]) is not list else element
        return pack([torch.tensor([_atom_name.index(ii) + 1 for ii in iele])
                     for iele in element])

    @classmethod
    def to_element(cls, number: Union[Tensor, List[Tensor]]):
        """Return elements number from elements."""
        if type(number) is Tensor:
            number = number.unsqueeze(0) if number.dim() == 1 else number
        return [[_atom_name[ii - 1] for ii in inum[inum.ne(0)]] for inum in number]

    @classmethod
    def from_ase_atoms(cls, atoms):
        """Instantiate a System instance from an ase.Atoms object.

        Arguments:
            atoms: ASE Atoms object(s) to be converted into System instance(s).

        Returns:
            System : System object.

        """
        if isinstance(atoms, list):  # If multiple atoms objects supplied:
            # Recursively call from_ase_atoms and return the result
            numbers = [torch.from_numpy(iat.numbers) for iat in atoms]
            positions = [torch.from_numpy(iat.positions) for iat in atoms]
            return System(numbers, positions)

        return System(torch.from_numpy(atoms.numbers),
                      torch.torch.from_numpy(atoms.positions))

    def to_hd5(self, target):
        """Convert the System instance to a set of hdf5 datasets.

        Return:
            target: The hdf5 entity to which the set of h5py.Dataset instances
                representing the system should be written.

        """
        # Short had for dataset creation
        add_data = target.create_dataset

        # Add datasets for numbers, positions, lattice, and pbc
        add_data('numbers', data=self.numbers)
        add_data('positions', data=self.positions.numpy())

    @staticmethod
    def from_hd5(source, **kwargs):
        """Convert an hdf5.Groups entity to a Systems instance.

        Arguments:
            source : hdf5 File/Group containing the system's data.

        Return:
            system : A systems instance representing the data stored.

        Notes:
            It should be noted that dtype will not be inherited from the
            database. Instead the default PyTorch dtype will be used.
        """
        # Get default dtype
        dtype = torch.get_default_dtype()

        # Read & parse datasets from the database into a System instance
        # & return the result.
        return System(
            torch.tensor(source['numbers']),
            torch.tensor(source['positions'], dtype=dtype), **kwargs)
