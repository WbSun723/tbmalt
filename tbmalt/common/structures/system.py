import torch
from typing import Union, List
import tbmalt.common.batch as batch
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
_l_num = {"H": 0, "He": 0,
          "Li": 0, "Be": 0,
          "B": 1, "C": 1, "N": 1, "O": 1, "F": 1, "Ne": 1,
          "Na": 0, "Mg": 0,
          "Al": 1, "Si": 1, "P": 1, "S": 1, "Cl": 1, "Ar": 1,
          "K": 0, "Ca": 0,
          "Sc": 2, "Ti": 2, "V": 2, "Cr": 2, "Mn": 2, "Fe": 2, "Co": 2,
          "Ni": 2, "Cu": 2, "Zn": 2,
          "Ga": 1, "Ge": 1, "As": 1, "Se": 1, "Br": 1, "Kr": 1,
          "Rb": 0, "Sr": 0,
          "Y": 2, "Zr": 2, "Nb": 2, "Mo": 2, "Tc": 2, "Ru": 2, "Rh": 2,
          "Pd": 2, "Ag": 2, "Cd": 2,
          "In": 1, "Sn": 1, "Sb": 1, "Te": 1, "I": 1, "Xe": 1,
          "Cs": 0, "Ba": 0,
          "La": 3, "Ce": 3, "Pr": 3, "Nd": 3, "Pm": 3, "Sm": 3, "Eu": 3,
          "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
          "Hf": 2, "Ta": 2, "W": 2, "Re": 2, "Os": 2, "Ir": 2, "Pt": 2,
          "Au": 2, "Hg": 2,
          "Tl": 1, "Pb": 1, "Bi": 1, "Po": 1, "At": 1}


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
            (`bool`, `array_like` [`bool`], optional)

    Notes:
        Units, such as distance, should be given in atomic units. PyTorch
        tensors will be instantiated using the default dtypes. These can be
        changed using torch.set_default_dtype.

    Todo:
        Add periodic boundaries.
    """

    def __init__(self, numbers: Union[Tensor, List[Tensor]],
                 positions: Union[Tensor, List[Tensor]],
                 lattice=None, **kwargs):
        self.unit = kwargs['unit'] if 'unit' in kwargs else 'angstrom'
        self.pbc = kwargs['pbc'] if 'pbc' in kwargs else lattice is not None
        self.positions, self.numbers = self._check(numbers, positions)

        # size of each system
        self.size_system = self._get_size()

        # get distance
        self.distances = self._get_distances()

        # get symbols
        self.symbols = self._get_symbols()

        # size of batch size, size of each system (number of atoms)
        self.size_batch = len(self.numbers)

        # get max l of each atom
        self.l_max = self._get_l_numbers()

        # orbital_index is the orbital index of each atom in each system
        # orbital_index_cumsum is the acculated orbital_index
        self.orbital_index, self.orbital_index_cumsum, self.number_orbital = \
            self._get_accumulated_orbital_numbers()

        # get Hamiltonian, overlap shape in batch
        self.hs_shape = self._get_hs_shape()

    def _check(self, numbers, positions):
        # sequences of tensor
        if type(numbers) is list:
            numbers = batch.pack(numbers)
        elif type(numbers) is torch.Tensor and numbers.dim() == 1:
            numbers = numbers.unsqueeze(0)

        # positions type check
        if type(positions) is list:
            positions = batch.pack(positions)
        elif type(positions) is torch.Tensor and positions.dim() == 2:
            positions = positions.unsqueeze(0)

        # transfer positions from angstrom to bohr
        positions = positions / _bohr if self.unit == 'angstrom' else positions
        return positions, numbers

    def _get_distances(self):
        """Return distances between a list of atoms for each system."""
        return batch.pack([torch.sqrt(((ipos[:inat].repeat(inat, 1) -
                                        ipos[:inat].repeat_interleave(inat, 0))
                                       ** 2).sum(1)).reshape(inat, inat)
                           for ipos, inat in zip(self.positions, self.size_system)])

    def _get_symbols(self):
        """Get atom name for each system in batch."""
        return [[_atom_name[ii - 1] for ii in inu[inu.ne(0.)]] for inu in self.numbers]

    def get_positions_vec(self):
        """Return positions vector between atoms."""
        return batch.pack([ipo.unsqueeze(-3) - ipo.unsqueeze(-2)
                           for ipo in self.positions])

    def _get_size(self):
        """Get each system size (number of atoms) in batch."""
        return [len(inum[inum.ne(0.)]) for inum in self.numbers]

    def _get_l_numbers(self):
        """Return the number of orbitals associated with each atom."""
        return [[_l_num[ii] for ii in isym] for isym in self.symbols]

    def _get_atom_orbital_numbers(self):
        """Return the number of orbitals associated with each atom.

        The atom orbital numbers is from (l + 1) ** 2.
        """
        return [[(_l_num[ii] + 1) ** 2 for ii in isym] for isym in self.symbols]

    def get_resolved_orbital(self):
        """Return l parameter and realted atom specie of each obital."""
        resolved_orbital_specie = \
            [sum([[ii] * int(jj) for ii, jj in zip(isym, iind)], [])
             for isym, iind in zip(self.symbols, self.orbital_index)]
        _l_orbital_res = [[0], [0, 1, 1, 1], [0, 1, 1, 1, 2, 2, 2, 2, 2]]
        l_orbital_res = [sum([_l_orbital_res[iil] for iil in il], [])
                         for il in self.l_max]
        return resolved_orbital_specie, l_orbital_res

    def _get_accumulated_orbital_numbers(self):
        """Return accumulated number of orbitals associated with each atom.

        For instance, for CH4, get_atom_orbital_numbers return [[4, 1, 1, 1,
        1]], this function will return [[0, 4, 5, 6, 7, 8]], max_orbital is 8.
        """
        atom_index = self._get_atom_orbital_numbers()
        index_cumsum = [torch.cumsum(torch.tensor(iind), 0).tolist()
                        for iind in atom_index]
        number_orbital = [int(ind[-1]) for ind in index_cumsum]
        return atom_index, index_cumsum, number_orbital

    def _get_hs_shape(self):
        """Return shapes of Hamiltonian and overlap."""
        maxorb = max([iorb[-1] for iorb in self.orbital_index_cumsum])

        # 3D shape: size of batch, total orbital number, total orbital number
        return torch.Size([len(self.orbital_index_cumsum), maxorb, maxorb])

    def get_global_species(self):
        """Get species for single or multi systems according to numbers."""
        numbers_ = torch.unique(self.numbers)
        numbers_nonzero = numbers_[numbers_.nonzero().squeeze()]
        if numbers_nonzero.dim() == 0:
            return _atom_name[numbers_nonzero - 1]
        else:
            return [_atom_name[ii - 1] for ii in numbers_nonzero]

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
            # return [cls.from_ase_atoms(iat) for iat in atoms]
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
    def from_hd5(source):
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
            torch.tensor(source['positions'], dtype=dtype))
