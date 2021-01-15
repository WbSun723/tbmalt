"""Load Slater-Koster Tables."""
import os
import numpy as np
import torch
import h5py
from scipy.interpolate import CubicSpline
from tbmalt.common.batch import pack
from tbmalt.common.maths.interpolator import SKInterpolation as SKInterp
Tensor = torch.Tensor
_orb = {1: 's', 6: 'p', 7: 'p', 8: 'p', 15: 'p', 16: 'p', 79: 'd'}


class IntegralGenerator:
    """Read skf files and return integrals.

    IntegralGenerator from_dir will read skf files with path of skf files and
    system object. Then return integrals according to parameters, such as
    distance, l_pair, etc.

    Argument:
        sktable_dict: This should be a dictionary of scipy splines keyed by a
            tuple of strings of the form: (z_1, z_2, ℓ_1, ℓ_2, b, O), where the
            ℓ's are the azimuthal quantum numbers, the z's are the atomic
            numbers, b is the bond type, & O is the operator which the spline
            represents i.e. S or H.
    """

    def __init__(self, sktable_dict):
        self.sktable_dict = sktable_dict

    @classmethod
    def from_dir(cls, path: str, system=None, elements=None, **kwargs) -> dict:
        """Read all skf files in a directory & return an SKIG instance.

        Arguments:
            path: path to the directory in which skf files can be found. If
                sk_type is normal, the path represents only the directory. if
                sk_type is h5py, the path is the joint directory and h5py file.
                sk_type is compression_radii, the path is the path to list of
                    SKF files with various compression radii.
            system: Optional, system object.
            element: Optional, global elements for read SKF files.

        Keyword Args:
            sk_type: type of skf files.
            sk_interpolation: interpolation method of integrals which are not
                in the grid points.
            orbital_resolve: If each orbital is resolved for U.
        """
        repulsive = kwargs.get('repulsive', False)

        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')

        sk_interp = kwargs.get('sk_interpolation', 'dftb_interpolation')

        # The interactions: ddσ, ddπ, ddδ, ...
        interactions = [(2, 2, 0), (2, 2, 1), (2, 2, 2), (1, 2, 0), (1, 2, 1),
                        (1, 1, 0), (1, 1, 1), (0, 2, 0), (0, 1, 0), (0, 0, 0)]

        # create a blank dict for integrals
        sktable_dict = {}
        assert elements is not None or system is not None

        # get global element species and all corresponding SKF files
        if system is not None:
            element, element_number, element_pair, element_number_pair = \
                system.get_global_species()

        # get global element species and all corresponding SKF files
        elif elements is not None:
            element_pair, element_number_pair = _get_element_info(elements)

        elif system is None and element is None:
            raise ValueError('At least one of system and element is not None')

        # loop of all global element pairs
        for ielement, ielement_number in zip(element_pair, element_number_pair):
            interpolator = SKInterp if sk_interp == 'dftb_interpolation' else CubicSpline

            # read skf files
            skf = LoadSKF.read(path, ielement, ielement_number, **kwargs)

            # generate skf files dict
            sktable_dict = _get_hs_dict(sktable_dict, interpolator, interactions, skf)

            if skf.homo:
                # return onsite
                if (*skf.elements.tolist(), 'onsite') not in sktable_dict:
                    sktable_dict = _get_onsite_dict(sktable_dict, skf)

                # retutn U
                if (*skf.elements.tolist(), 'U') not in sktable_dict:
                    sktable_dict = _get_u_dict(sktable_dict, skf, **kwargs)

            if repulsive:
                sktable_dict = _get_repulsive_dict(sktable_dict, skf)

        return cls(sktable_dict)

    def __call__(self, distances: Tensor, atom_pair: Tensor,
                 l_pair: Tensor, **kwargs) -> Tensor:
        """Get integrals for given systems.

        Arguments:
            distances: distances of single & multi systems.
            atom_pair: skf files type. Support normal skf, h5py binary skf.
            l_pair:
        Keyword Args:
            hs_type: type of skf files.
            sk_interpolation: interpolation method of integrals which are not
                in the grid points.
            orbital_resolve: If each orbital is resolved for U.
        """
        if kwargs.get('orbital_resolve', False):
            raise NotImplementedError('Not implement orbital resolved U.')
        hs_type = kwargs.get('hs_type', 'H')

        # Retrieve the appropriate splines
        splines = [self.sktable_dict[(
            *atom_pair.tolist(), *l_pair.tolist(), b, hs_type)]
            for b in range(min(l_pair) + 1)]

        list_integral = [spline(distances) for spline in splines]
        if type(list_integral[0]) == np.ndarray:
            list_integral = [torch.from_numpy(ii) for ii in list_integral]

        return pack(list_integral).T

    def get_onsite(self, atom_number: Tensor) -> Tensor:
        """Return onsite with onsite_blocks.

        Arguments:
            atom_number: Atomic numbers.
        """
        return torch.cat([self.sktable_dict[
            (*[ii.tolist(), ii.tolist()], 'onsite')] for ii in atom_number])

    def get_U(self, atom_number: Tensor) -> Tensor:
        """Return onsite with onsite_blocks.

        Arguments:
            atom_number: Atomic numbers.
        """
        return torch.cat([self.sktable_dict[
            (*[ii.tolist(), ii.tolist()], 'U')] for ii in atom_number])

    def get_repulsive(self) -> dict:
        """Return repulsive parameter."""
        return self.sktable_dict


def _get_element_info(elements: list):
    """Generate element pair information."""
    _elements_dict = {"H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
                      "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
                      "S": 16, "Ti": 22, "Au": 79}
    element_pair = [[iel, jel] for iel, jel in zip(
        sorted(elements * len(elements)), elements * len(elements))]
    element_number_pair = [[_elements_dict[ii[0]], _elements_dict[ii[1]]]
                           for ii in element_pair]
    return element_pair, element_number_pair


def _get_hs_dict(sktable_dict: dict, interpolator: object,
                 interactions: list, skf: object) -> dict:
    """Get sk tables for each orbital interaction.

    Arguments:
        sktable_dict: sk tables dictionary.
        interpolator: sk interpolation method.
        interactions: orbital interactions, e.g. (0, 0, 0) for ss0 orbital.
        skf: object with SKF integrals data.
    """
    for ii, name in enumerate(interactions):
        sktable_dict[(*skf.elements.tolist(), *name, 'H')] = \
            interpolator(skf.hs_grid, skf.hamiltonian.T[ii])
        sktable_dict[(*skf.elements.tolist(), *name, 'S')] = \
            interpolator(skf.hs_grid, skf.overlap.T[ii])
    return sktable_dict


def _get_onsite_dict(sktable_dict: dict, skf: object) -> dict:
    """Get sk tables for global element.

    Arguments:
        sktable_dict: sk tables dictionary.
        skf: object with SKF integrals data.
    """
    # add onsite, return onsite obey sequence s, p, d ...
    onsite = skf.onsite.unsqueeze(1)

    if _orb[skf.elements.tolist()[0]] == 's':
        sktable_dict[(*skf.elements.tolist(), 'onsite')] = torch.cat([onsite[-1]])

    elif _orb[skf.elements.tolist()[0]] == 'p':
        sktable_dict[(*skf.elements.tolist(), 'onsite')] = \
            torch.cat([onsite[-1], onsite[-2].repeat(3)])

    elif _orb[skf.elements.tolist()[0]] == 'd':
        sktable_dict[(*skf.elements.tolist(), 'onsite')] = \
            torch.cat([onsite[-1], onsite[-2].repeat(3), onsite[-3].repeat(5)])
    return sktable_dict


def _get_u_dict(sktable_dict: dict, skf: object, **kwargs) -> dict:
    """Get sk tables for global element.

    Arguments:
        sktable_dict: sk tables dictionary.
        skf: object with SKF integrals data.
    """
    if kwargs.get('orbital_resolve', False):
        raise NotImplementedError('Not implement orbital resolved U.')

    # add onsite, return onsite obey sequence s, p, d ...
    sktable_dict[(*skf.elements.tolist(), 'U')] = skf.U.unsqueeze(1)[-1]
    return sktable_dict


def _get_repulsive_dict(sktable_dict: dict, skf: object) -> dict:
    """Return repulsive parameter."""
    sktable_dict[(*skf.elements.tolist(), 'n_repulsive')] = skf.n_repulsive
    sktable_dict[(*skf.elements.tolist(), 'rep_cutoff')] = skf.rep_cutoff
    sktable_dict[(*skf.elements.tolist(), 'rep_table')] = skf.rep_table
    sktable_dict[(*skf.elements.tolist(), 'rep_grid')] = skf.rep_grid
    sktable_dict[(*skf.elements.tolist(), 'rep_short')] = skf.rep_short
    sktable_dict[(*skf.elements.tolist(), 'rep_long_c')] = skf.rep_long_c
    sktable_dict[(*skf.elements.tolist(), 'rep_long_grid')] = skf.rep_long_grid
    return sktable_dict


class LoadSKF:
    """Get integrals for given systems.

        Arguments:
            elements: global elements of single & multi systems.
            hamiltonian: Hamiltonian skf table data.
            overlap: Overlap skf table data.

        Keyword Args:
            hs_type: type of skf files.
            sk_interpolation: interpolation method of integrals which are not
                in the grid points.
    """

    def __init__(self, elements, hamiltonian: Tensor, overlap: Tensor,
                 hs_grid: Tensor, hs_cutoff, **kwargs):
        repulsive = kwargs.get('repulsive', False)
        # If a single element was specified, resolve it to a tensor
        if isinstance(elements, int):
            self.elements = torch.tensor([elements]*2)
        elif isinstance(elements, list):
            self.elements = torch.tensor(elements)
        else:
            self.elements = elements

        # homo or hetero
        self.homo = self.elements[0] == self.elements[1]
        if self.homo:
            self.onsite = kwargs['onsite']
            self.U = kwargs['U']

        self.hamiltonian = hamiltonian
        self.overlap = overlap
        self.hs_grid = hs_grid

        # Repulsion properties
        self.hs_cutoff = hs_cutoff

        if repulsive:
            self.n_repulsive = kwargs['n_repulsive']
            self.rep_cutoff = kwargs['rep_cutoff']
            self.rep_table = kwargs['rep_table']
            self.rep_grid = kwargs['rep_grid']
            self.rep_long_grid = kwargs['rep_long_grid']
            self.rep_short = kwargs['rep_short']
            self.rep_long_c = kwargs['rep_long_c']

        # for polynomial representation of
        self.rep_poly = kwargs.get('r_poly', None)

        # identify the skf specification version
        if 'version' in kwargs:
            self.version = kwargs['version']

    @classmethod
    def read(cls, path: str, element: list, element_number: list, **kwargs):
        """Read different type SKF files.

        Arguments:
            path: path to SKF files.
            sk_type: type of SKF files.
            element: element pair of SKF files.
            element_number: pair element numbers.
        """
        sk_type = kwargs.get('sk_type', 'normal')
        if sk_type == 'normal':
            path = os.path.join(path, element[0] + '-' + element[1] + '.skf')
            return cls.read_normal(path, element_number, **kwargs)
        elif sk_type == 'h5py':
            return cls.read_hdf(path, element, element_number, **kwargs)
        elif sk_type == 'compression_radii':
            return cls.read_compression_radii(path, **kwargs)

    @classmethod
    def _get_version_number(cls, file: str, lines: list) -> str:
        """Return skf version, see: https://github.com/bhourahine/slako-doc."""
        if file.startswith('@'):  # If 1'st char is @, the version is 1.0e
            return '1.0e'
        elif len(lines[0].split()) == 2 and lines[0].split()[1].isnumeric():
            return '0.9'  # If no comment line; this must be version 0.9

    @classmethod
    def _asterisk_to_repeat_tensor(cls, xx: list) -> Tensor:
        """Transfer 'int*float' to repetitive tensor float.repeat(int)."""
        return torch.cat([torch.tensor([float(ii.split('*')[1])]).repeat(
            int(ii.split('*')[0])) if '*' in ii else torch.tensor([float(ii)])
            for ii in xx])

    @classmethod
    def read_normal(cls, path_to_skf: str, element_number: list, **kwargs):
        """Read in a skf file and returns an SKF_File instance.

        File names should follow the naming convention X-Y.skf where X and
        Y are the chemical symbol's of the elements involved.

        Arguments:
            path_to_skf: Path to the target skf file.
            element_number: Input element number pair.
        """
        repulsive = kwargs.get('repulsive', False)

        # alias for common code structure
        lmf = lambda xx: list(map(float, xx.split()))
        _asterisk = lambda xx: list(map(str.strip, xx.split()))

        file = open(path_to_skf, 'r').read()
        lines = file.split('\n')

        ver = cls._get_version_number(file, lines)
        lines = lines[1:] if ver in ['1.0e'] else lines

        # Get atomic numbers & identify if this is the homo case
        homo = element_number[0] == element_number[1]
        is_asterisk = '*' in lines[2] or '*' in lines[2 + homo]

        if homo:
            if is_asterisk:
                homo_ln = cls._asterisk_to_repeat_tensor(_asterisk(
                    lines[1].replace(',', ' ')))
            else:
                homo_ln = torch.tensor(lmf(lines[1]))
            n = int((len(homo_ln) - 1) / 3)  # <- Number of shells specified
            onsite, _, U, occupations = homo_ln.split_with_sizes([n, 1, n, n])

        # grid distance and grid points number
        g_step, n_points = lines[0].replace(',', ' ').split()[:2]
        g_step, n_points = float(g_step), int(n_points)

        # construct distance list. Note; distance start at 1 * g_step not zero
        hs_grid = torch.arange(1, n_points + 1) * g_step

        # Fetch the H and S sk tables
        if is_asterisk:
            hamiltonian, overlap = torch.stack([
                cls._asterisk_to_repeat_tensor(_asterisk(ii.replace(',', ' ')))
                for ii in lines[2 + homo: 2 + n_points + homo]]).chunk(2, 1)
        else:
            hamiltonian, overlap = torch.tensor(
                [lmf(ii) for ii in lines[2 + homo: 2 + n_points + homo]]).chunk(2, 1)

        # 2 + homo line data, which contains mass, rcut and cutoff
        if is_asterisk:
            mass, *r_poly, cutoff = cls._asterisk_to_repeat_tensor(_asterisk(
                lines[2 + homo].replace(',', '')))[:10]
        else:
            mass, *r_poly, cutoff = torch.tensor(lmf(lines[2 + homo]))[:10]

        # Check if there is a spline representation
        if 'Spline' in file and repulsive:
            start = lines.index('Spline') + 1  # Identify spline section start

            # get number of spline sections & overwrite the r_cutoff previously
            nrep, r_cutoff = lines[start].split()
            nrep, r_cutoff = int(nrep), float(r_cutoff)

            r_tab = torch.tensor([lmf(line) for line in
                                  lines[start + 2: start + 1 + nrep]])

            rep = r_tab[:, 2:]  # -> repulsive tables
            r_grid = torch.tensor([*r_tab[:, 0], r_tab[-1, 1]])

            # Get the short and long range terms
            r_short = torch.tensor(lmf(lines[start + 1]))
            r_long_tab = torch.tensor(lmf(lines[start + 1 + nrep]))
            r_long_grid = r_long_tab[:2]
            r_long_c = r_long_tab[2:]

        # Build the parameter lists to pass to the in_grid_pointst method
        pos = (element_number, hamiltonian, overlap, hs_grid, cutoff)

        # Those that are passed by keyword
        kwd = {'version': ver, 'r_poly': r_poly}

        if homo:  # Passed only if homo case
            kwd.update({'mass': mass, 'onsite': onsite, 'U': U,
                        'occupations': occupations})

        if 'Spline' in file and repulsive:
            kwd.update({'n_repulsive': nrep, 'rep_table': rep,
                        'rep_grid': r_grid, 'rep_short': r_short,
                        'rep_long_grid': r_long_grid, 'rep_long_c': r_long_c,
                        'rep_cutoff': r_cutoff, 'repulsive': True})

        return cls(*pos, **kwd)

    @classmethod
    def read_hdf(cls, path: str, element_pair: list, element_number: list,
                 **kwargs):
        """Generate integral from h5py binary data."""
        repulsive = kwargs.get('repulsive', False)

        element_ij = element_pair[0] + element_pair[1]
        if not os.path.isfile(path):
            raise FileExistsError('dataset %s do not exist' % path)

        kwd = {}  # create empty dict
        with h5py.File(path, 'r') as f:
            hs_grid = f[element_ij + '/hs_grid'][()]
            hs_cutoff = f[element_ij + '/hs_cutoff'][()]
            hamiltonian = torch.from_numpy(f[element_ij + '/hamiltonian'][()])
            overlap = torch.from_numpy(f[element_ij + '/overlap'][()])

            # return hamiltonian, overlap, and related data
            pos = (element_number, hamiltonian, overlap, hs_grid, hs_cutoff)

            # homo SKF files
            if element_pair[0] == element_pair[1]:
                onsite = torch.from_numpy(f[element_ij + '/onsite'][()])
                U = torch.from_numpy(f[element_ij + '/U'][()])
                kwd.update({'onsite': onsite, 'U': U})

            if repulsive:
                nrep = f[element_ij + '/n_repulsive'][()]
                rep = torch.from_numpy(f[element_ij + '/rep_table'][()])
                r_grid = torch.from_numpy(f[element_ij + '/rep_grid'][()])
                r_short = torch.from_numpy(f[element_ij + '/rep_short'][()])
                r_long_grid = torch.from_numpy(f[element_ij + '/rep_long_grid'][()])
                r_long_c = torch.from_numpy(f[element_ij + '/rep_long_c'][()])
                r_cutoff = f[element_ij + '/rep_cutoff'][()]
                kwd.update({'n_repulsive': nrep, 'rep_table': rep,
                            'rep_grid': r_grid, 'rep_short': r_short,
                            'rep_long_grid': r_long_grid, 'repulsive': True,
                            'rep_cutoff': r_cutoff, 'rep_long_c': r_long_c})

        return cls(*pos, **kwd)

    @classmethod
    def read_compression_radii(cls, path, **kwargs):
        """Read in a skf file and returns an SKF_File instance."""
        raise NotImplementedError()


