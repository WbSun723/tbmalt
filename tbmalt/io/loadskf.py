"""Load Slater-Koster Tables."""
import os
import numpy as np
import torch
import h5py
from scipy.interpolate import CubicSpline
from tbmalt.common.batch import pack
_orb = {1: 's', 6: 'p', 7: 'p', 8: 'p', 79: 'd'}


class IntegralGenerator:
    """Read skf files and return integrals.

    IntegralGenerator from_dir will read skf files with path of skf files. Then
    return integrals according to parameters, such as distance, l_pair, etc.

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
    def from_dir(cls, path, system=None, elements=None, **kwargs):
        """Read all skf files in a directory & return an SKIG instance.

        Argument:
            path: path to the directory in which skf files can be found. If
                sk_type is normal, the path represents only the directory. if
                sk_type is h5py, the path is the joint directory and h5py file.
            system: system object.

        Keyword Args:
            sk_type: type of skf files.
            sk_interpolation: interpolation method of integrals which are not
                in the grid points.
        """
        sk_type = kwargs.get('sk_type', 'normal')
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
        elif elements is not None:
            element_pair, element_number_pair = _get_element_info(elements)

        # loop of all global element pairs
        for ielement, ielement_number in zip(element_pair, element_number_pair):
            interpolator = DFTBInterpolation \
                if sk_interp == 'dftb_interpolation' else CubicSpline

            # read skf files
            skf = _read_skf(path, sk_type, ielement, ielement_number)

            # generate skf files dict
            sktable_dict = _get_sk_dict(
                sktable_dict, interpolator, interactions, skf)

        return cls(sktable_dict)

    def __call__(self, distances, atom_number_pair, l_pair, **kwargs):
        """Get integrals for given systems.

        Argument:
            distances: distances of single & multi systems.
            atom_number_pair: all element number pairs.
            l_pair: all l number pairs.

        Keyword Args:
            hs_type: type of skf files.
            sk_interpolation: interpolation method of integrals which are not
                in the grid points.
        """
        hs_type = kwargs.get('hs_type', 'H')

        # Retrieve the appropriate splines
        splines = [self.sktable_dict[(
            *atom_number_pair.tolist(), *l_pair.tolist(), b, hs_type)]
            for b in range(min(l_pair) + 1)]

        list_integral = [spline(distances) for spline in splines]
        if type(list_integral[0]) == np.ndarray:
            list_integral = [torch.from_numpy(ii) for ii in list_integral]

        return pack(list_integral).T

    def get_onsite(self, onsite_blocks):
        """Return onsite."""
        return torch.cat([self.sktable_dict[
            (*[ii.tolist(), ii.tolist()], 'onsite')] for ii in onsite_blocks])


def _get_element_info(elements):
    """Generate element pair information."""
    _elements_dict = {"H": 1, "C": 6, "N": 7, "O": 8, "Au": 79}
    element_pair = [[iel, jel] for iel, jel in zip(
        sorted(elements * len(elements)), elements * len(elements))]
    element_number_pair = [[_elements_dict[ii[0]], _elements_dict[ii[1]]]
                           for ii in element_pair]
    return element_pair, element_number_pair


def _read_skf(path, sk_type, element, element_number):
    """Read different type SKF files.

    Argument:
        path: path to SKF files.
        sk_type: type of SKF files.
        element: element pair of SKF files.
        element_number: pair element numbers.
    """
    if sk_type == 'normal':
        isk = os.path.join(path, element[0] + '-' + element[1] + '.skf')
        return LoadSKF.read(isk, element_number=element_number)
    elif sk_type == 'h5py':
        return LoadSKF.read(path, sk_type='h5py',
                            element=element, element_number=element_number)


def _get_sk_dict(sktable_dict, interpolator, interactions, skf):
    """Get sk tables for each orbital interaction.

    Argument:
        sktable_dict: sk tables dictionary.
        interpolator: sk interpolation method.
        interactions: orbital interactions, such as (0, 0, 0), or ss0 orbital.
        skf: object with SKF integrals data.
    """
    for ii, name in enumerate(interactions):
        sktable_dict[(*skf.elements.tolist(), *name, 'H')] = \
            interpolator(skf.hs_grid, skf.hamiltonian.T[ii])
        sktable_dict[(*skf.elements.tolist(), *name, 'S')] = \
            interpolator(skf.hs_grid, skf.overlap.T[ii])

        # add onsite, return onsite obey sequence s, p, d ...
        if skf.homo:
            onsite = skf.onsite.unsqueeze(1)
            if _orb[skf.elements.tolist()[0]] == 's':
                sktable_dict[(*skf.elements.tolist(), 'onsite')] = \
                    torch.cat([onsite[-1]])
            elif _orb[skf.elements.tolist()[0]] == 'p':
                sktable_dict[(*skf.elements.tolist(), 'onsite')] = \
                    torch.cat([onsite[-1], onsite[-2].repeat(3)])
            elif _orb[skf.elements.tolist()[0]] == 'd':
                sktable_dict[(*skf.elements.tolist(), 'onsite')] = \
                    torch.cat([onsite[-1], onsite[-2].repeat(3),
                               onsite[-3].repeat(5)])
    return sktable_dict


class LoadSKF:
    """Get integrals for given systems.

        Argument:
            elements: global elements of single & multi systems.
            hamiltonian: hamiltonian in SK tables.
            overlap: overlap in SK tables.
            hs_grid: grid distances of hamiltonian and overlap.
            R_cutoff: cutoff of hamiltonian and overlap.

        Keyword Args:
            hs_type: type of skf files.
            sk_interpolation: interpolation method of integrals which are not
                in the grid points.
    """
    def __init__(self, elements, hamiltonian, overlap, hs_grid, R_cutoff, **kwargs):
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

        self.hamiltonian = hamiltonian
        self.overlap = overlap
        self.hs_grid = hs_grid

        # Repulsion properties
        self.R_cutoff = R_cutoff
        if 'repulsive' in kwargs:
            try:
                self.repulsive = kwargs['repulsive']
                self.r_short = kwargs['R_short']
                self.r_long = kwargs['R_long']
                self.r_grid = kwargs['R_grid']
            except KeyError as e:
                raise KeyError(
                    f'Repulsive spline missing "{e.args[0]}" keyword argument.')

        # For polynomial representation
        self.r_poly = kwargs.get('r_poly', None)
        self.__repulsion_by_spline = self.r_poly is None

        # Identify the skf specification version
        if 'version' in kwargs:
            self.version = kwargs['version']
        elif self.hamiltonian.shape[1] == 20:
            self.version = '1.0e'
        else:
            self.version = '1.0'

    @property
    def repulsion_by_spline(self):
        """True if a repulsive spline present & False if polynomial used."""
        # Getter used to prevent manual alteration
        return self.__repulsion_by_spline

    @classmethod
    def read(cls, path, system=None, sk_type='normal', **kwargs):
        if not os.path.exists(path):
            raise FileNotFoundError('Target path does not exist')
        if sk_type == 'normal':
            return cls._read_normal(path, kwargs['element_number'])
        elif sk_type == 'h5py':
            return cls._read_hdf(path, system, kwargs['element'], kwargs['element_number'])

    @classmethod
    def get_version_number(cls, file, lines):
        """Return skf version number."""
        if file.startswith('@'):  # If 1'st char is @, the version is 1.0e
            v = '1.0e'
        elif len(lines[0].split()) == 2 and lines[0].split()[1].isnumeric():
            v = '0.9'  # If no comment line; this must be version 0.9
        else:  # Otherwise version 1.0
            v = '1.0'
        return v

    @classmethod
    def _read_normal(cls, path, element_number, system=None):
        """Read in a skf file and returns an SKF_File instance.

        File names should follow the naming convention X-Y.skf where X and
        Y are the chemical symbol's of the elements involved.

        Arguments:
            path: Path to the target skf file.
        """
        # Alias for common code structure; convert str to list of floats
        lmf = lambda x: list(map(float, x.split()))

        file = open(path, 'r').read()
        lines = file.split('\n')

        ver = cls.get_version_number(file, lines)
        lines = lines[1:] if ver in ['1.0', '1.0e'] else lines

        # Get atomic numbers & identify if this is the homo case
        homo = element_number[0] == element_number[1]

        if homo:
            # & occupations
            homo_ln = torch.tensor(lmf(lines[1]))
            n = int((len(homo_ln) - 1) / 3)  # <- Number of shells specified
            onsite, _, U, occupations = homo_ln.split_with_sizes([n, 1, n, n])

        # H & S Tables
        g_step, n_points = lines[0].split()
        g_step, n_points = float(g_step), int(n_points)

        # Construct distance list. Note; distance start at 1 * g_step not zero
        hs_grid = torch.arange(1, n_points + 1) * g_step

        # Fetch the H and S sk tables
        hamiltonian, overlap = torch.tensor(
            [lmf(i) for i in lines[2 + homo: 2 + n_points + homo]]).chunk(2, 1)

        # Repulsive Data
        mass, *R_poly, r_cutoff = torch.tensor(lmf(lines[2 + homo]))[:10]

        # Check if there is a spline representation
        has_r_spline = 'Spline' in file

        if has_r_spline:
            start = lines.index('Spline') + 1  # Identify spline section start

            # Read number of spline sections & overwrite the r_cutoff previously
            # fetched from the polynomial line.
            n, r_cutoff = lines[start].split()
            n, r_cutoff = int(n), float(r_cutoff)

            r_tab = torch.tensor([lmf(line) for line in lines[start + 2: start + 1 + n]])

            R = r_tab[:, 2:]
            R_grid = torch.tensor([*r_tab[:, 0], r_tab[-1, 1]])

            # Get the short and long range terms
            R_short = torch.tensor(lmf(lines[start + 1]))
            R_long = torch.tensor(lmf(lines[start + 1 + n])[3:])

        # Build the parameter lists to pass to the in_grid_pointst method
        pos = (element_number, hamiltonian, overlap, hs_grid, r_cutoff)

        # Those that are passed by keyword
        kwd = {'version': ver}

        if homo:  # Passed only if homo case
            kwd.update({'mass': mass, 'onsite': onsite, 'U': U,
                        'occupations': occupations})

        if has_r_spline:  # Passed only if there is a repulsive spline
            kwd.update({'R': R, 'R_grid': R_grid, 'R_short': R_short,
                        'R_long': R_long})

        else:  # Passed if there is no repulsive spline
            kwd.update({'R_poly': R_poly})

        return cls(*pos, **kwd)

    @classmethod
    def _read_hdf(cls, path, system, element_pair, element_number):
        """Generate integral along distance dimension."""
        if not os.path.isfile(path):
            raise FileExistsError('dataset %s do not exist' % path)

        with h5py.File(path, 'r') as f:
            # get the grid sidtance, which should be the same
            g_step = f[element_pair[0] + element_pair[1] + '/grid_dist'][()]
            n_points = f[element_pair[0] + element_pair[1] + '/ngridpoint'][()]
            hs_grid = torch.arange(1, n_points + 1) * g_step

            # get hamiltonian and overlap
            hs = torch.from_numpy(f[element_pair[0] + element_pair[1] + '/hs_all'][()])
            r_cutoff = (n_points + 1) * g_step
            hamiltonian, overlap = hs[:, :10], hs[:, 10:]
            pos = (element_number, hamiltonian, overlap, hs_grid, r_cutoff)
        return cls(*pos)

    @classmethod
    def _read_compression_radii(cls, path):
        """Read in a skf file and returns an SKF_File instance."""
        pass

    @classmethod
    def from_splines(cls):
        """Instantiates an ``SKF_File`` entity from a set of splines."""
        raise NotImplementedError()


class DFTBInterpolation:
    """Interpolation of SK tables.

    Arguments:
        x: Grid points of distances.
        y: integral tables.
    """

    def __init__(self, x, y):
        self.incr = x[1] - x[0]
        self.y = y
        self.ngridpoint = len(x)

    def __call__(self, rr, ninterp=8, delta_r=1E-5, ntail=5):
        """Interpolation SKF according to distance from integral tables.

        Arguments:
            rr: interpolation points
            ngridpoint: number of total grid points
            distance between atoms
            ninterp: interpolate from up and lower SKF grid points number

        """
        tail = ntail * self.incr
        rmax = (self.ngridpoint + ntail - 1) * self.incr
        ind = (rr / self.incr).int()
        result = torch.zeros(rr.shape)

        # thye input SKF must have more than 8 grid points
        if self.ngridpoint < ninterp + 1:
            raise ValueError("Not enough grid points for interpolation!")

        # distance beyond grid points in SKF
        if (rr >= rmax).any():
            result[rr >= rmax] = 0.

        # => polynomial fit
        elif (ind <= self.ngridpoint).any():
            _mask = ind <= self.ngridpoint

            # get the index of rr in grid points
            ind_last = (ind[_mask] + ninterp / 2 + 1).int()
            ind_last[ind_last > self.ngridpoint] = self.ngridpoint
            ind_last[ind_last < ninterp] = ninterp

            xa = (ind_last.unsqueeze(1) - ninterp + torch.arange(ninterp)) * self.incr
            yb = torch.stack([self.y[ii - ninterp - 1: ii - 1] for ii in ind_last])
            result[_mask] = self.poly_interp_2d(xa, yb, rr)

        # Beyond the grid => extrapolation with polynomial of 5th order
        elif (self.ngridpoint < ind < self.ngridpoint + ntail - 1).any():
            dr = rr - rmax
            ilast = self.ngridpoint
            xa = (ilast - ninterp + torch.arange(ninterp)) * self.incr
            yb = self.y[ilast - ninterp - 1: ilast - 1]
            y0 = self.poly_interp_2d(xa, yb, xa[ninterp - 1] - delta_r)
            y2 = self.poly_interp_2d(xa, yb, xa[ninterp - 1] + delta_r)
            ya = self.y[ilast - ninterp - 1: ilast - 1]
            y1 = ya[ninterp - 1]
            y1p = (y2 - y0) / (2.0 * delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
            dd = self.poly5_zero(y1, y1p, y1pp, dr, -1.0 * tail)
        return result

    def poly5_zero(self, y0, y0p, y0pp, xx, dx):
        """Get integrals if beyond the grid range with 5th polynomial."""
        dx1 = y0p * dx
        dx2 = y0pp * dx * dx
        dd = 10.0 * y0 - 4.0 * dx1 + 0.5 * dx2
        ee = -15.0 * y0 + 7.0 * dx1 - 1.0 * dx2
        ff = 6.0 * y0 - 3.0 * dx1 + 0.5 * dx2
        xr = xx / dx
        yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr
        return yy

    def poly_interp_2d(self, xp, yp, rr):
        """Interpolate from DFTB+ (lib_math) with uniform grid.

        Arguments:
            xp: 2D tensor, 1st dimension if batch size, 2nd is grid points.
            yp: 2D tensor of integrals.
            rr: interpolation points.
        """
        nn0, nn1 = xp.shape[0], xp.shape[1]
        index_nn0 = torch.arange(nn0)
        icl = torch.zeros(nn0).long()
        cc, dd = torch.zeros(yp.shape), torch.zeros(yp.shape)

        cc[:], dd[:] = yp[:], yp[:]
        dxp = abs(rr - xp[index_nn0, icl])

        # find the most close point to rr (single atom pair or multi pairs)
        _mask, ii = torch.zeros(len(rr)) == 0, 0
        dxNew = abs(rr - xp[index_nn0, 0])
        while (dxNew < dxp).any():
            ii += 1
            assert ii < nn1 - 1  # index ii range from 0 to nn1 - 1
            _mask = dxNew < dxp
            icl[_mask] = ii
            dxp[_mask] = abs(rr - xp[index_nn0, ii])[_mask]

        yy = yp[index_nn0, icl]
        for mm in range(nn1 - 1):
            for ii in range(nn1 - mm - 1):
                rtmp0 = xp[index_nn0, ii] - xp[index_nn0, ii + mm + 1]
                rtmp1 = (cc[index_nn0, ii + 1] - dd[index_nn0, ii]) / rtmp0
                cc[index_nn0, ii] = (xp[index_nn0, ii] - rr) * rtmp1
                dd[index_nn0, ii] = (xp[index_nn0, ii + mm + 1] - rr) * rtmp1
            if (2 * icl < nn1 - mm - 1).any():
                _mask = 2 * icl < nn1 - mm - 1
                yy[_mask] = (yy + cc[index_nn0, icl])[_mask]
            else:
                _mask = 2 * icl >= nn1 - mm - 1
                yy[_mask] = (yy + dd[index_nn0, icl - 1])[_mask]
                icl[_mask] = icl[_mask] - 1
        return yy
