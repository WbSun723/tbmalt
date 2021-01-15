import bisect
import torch
Tensor = torch.Tensor


class BicubInterp:
    """Vectorized bicubic interpolation method designed for molecule.

    The bicubic interpolation is designed to interpolate the integrals of
    whole molecule. The xmesh, ymesh are the grid points and they are the same.
    The zmesh is a 4D or 5D Tensor. The 1st, 2nd dimensions are corresponding
    to the pairwise atoms in molecule. The 3rd and 4th are corresponding to
    the xmesh and ymesh. The 5th dimension is optional. For bicubic
    interpolation of single integral such as ss0 orbital, it is 4D Tensor.
    For bicubic interpolation of all the orbital integral, zmesh is 5D Tensor.

    Arguments:
        xmesh: a 1D Tensor
        zmesh: a 4D or 5D Tensor

    Reference:
        .. [Bicubic wiki] https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self, xmesh: Tensor, zmesh: Tensor):
        """Get interpolation with two variables."""
        self.xmesh = xmesh
        self.zmesh = zmesh

    def __call__(self, xnew: Tensor):
        """Calculate bicubic interpolation.

        Arguments:
            xnew: the interpolation points of each atom in molecule
        """
        self.xi = xnew
        self.nat = len(self.xi)
        coeff = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                              [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = torch.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                               [0., 1., -2., 1.], [0., 0., -1., 1.]])

        # get the nearest grid points indices of xi
        self._get_indices()

        # this is to transfer x to fraction
        xf = (self.xi - self.xmesh.T[self.nx0, self.nind]) / (self.xmesh.T[
            self.nx1, self.nind] - self.xmesh.T[self.nx0, self.nind])
        xmat = torch.stack([xf ** 0, xf ** 1, xf ** 2, xf ** 3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self._fmat0th(self.zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = \
            self._fmat1th(self.zmesh, f00, f10, f01, f11)
        fmat = torch.stack([torch.stack([f00, f01, f02, f03]),
                            torch.stack([f10, f11, f12, f13]),
                            torch.stack([f20, f21, f22, f23]),
                            torch.stack([f30, f31, f32, f33])])

        pdim = [2, 3, 0, 1] if fmat.dim() == 4 else [2, 3, 4, 0, 1]
        a_mat = torch.matmul(torch.matmul(coeff, fmat.permute(pdim)), coeff_)
        return torch.stack([torch.stack(
            [torch.matmul(torch.matmul(xmat[:, i], a_mat[i, j]), xmat[:, j])
             for j in range(self.nat)]) for i in range(self.nat)])

    def _get_indices(self):
        """Get indices and repeat indices."""
        xi = self.xi.cpu() if self.xi.device.type == 'cuda' else self.xi
        xm = self.xmesh.cpu() if self.xmesh.device.type == 'cuda' else self.xmesh
        self.nx0 = torch.tensor([
            bisect.bisect(xm[ii].detach().numpy(), xi[ii].detach().numpy()) - 1
            for ii in range(self.nat)])
        self.nx0ri = self.nx0.repeat_interleave(self.nat)
        self.nx0r = self.nx0.repeat(self.nat)

        # get all surrounding 4 grid points indices and repeat indices
        self.nind = torch.tensor([ii for ii in range(self.nat)])
        self.nindri = self.nind.repeat_interleave(self.nat)
        self.nindr = self.nind.repeat(self.nat)
        self.nx1 = torch.tensor([ii + 1 for ii in self.nx0])
        self.nx1ri = self.nx1.repeat_interleave(self.nat)
        self.nx1r = self.nx1.repeat(self.nat)
        self.nx_1 = torch.tensor([ii - 1 if ii >= 1 else ii for ii in self.nx0])
        self.nx_1ri = self.nx_1.repeat_interleave(self.nat)
        self.nx_1r = self.nx_1.repeat(self.nat)
        self.nx2 = torch.tensor([ii + 2 if ii <= len(self.xmesh) - 3
                                 else ii + 1 for ii in self.nx0])
        self.nx2ri = self.nx2.repeat_interleave(self.nat)
        self.nx2r = self.nx2.repeat(self.nat)

    def _fmat0th(self, zmesh):
        """Construct f(0/1, 0/1) in fmat."""
        f00 = zmesh[self.nindri, self.nindr, self.nx0ri, self.nx0r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f10 = zmesh[self.nindri, self.nindr, self.nx1ri, self.nx0r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f01 = zmesh[self.nindri, self.nindr, self.nx0ri, self.nx1r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f11 = zmesh[self.nindri, self.nindr, self.nx1ri, self.nx1r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        return f00, f10, f01, f11

    def _fmat1th(self, zmesh, f00, f10, f01, f11):
        """Get the 1st derivative of four grid points over x, y and xy."""
        f_10 = zmesh[self.nindri, self.nindr, self.nx_1ri, self.nx0r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f_11 = zmesh[self.nindri, self.nindr, self.nx_1ri, self.nx_1r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f0_1 = zmesh[self.nindri, self.nindr, self.nx0ri, self.nx_1r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f02 = zmesh[self.nindri, self.nindr, self.nx0ri, self.nx2r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f1_1 = zmesh[self.nindri, self.nindr, self.nx_1ri, self.nx_1r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f12 = zmesh[self.nindri, self.nindr, self.nx1ri, self.nx2r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f20 = zmesh[self.nindri, self.nindr, self.nx2ri, self.nx0r].reshape(
            self.nat, self.nat, -1).squeeze(-1)
        f21 = zmesh[self.nindri, self.nindr, self.nx2ri, self.nx1r].reshape(
            self.nat, self.nat, -1).squeeze(-1)

        # calculate the derivative: (F(1) - F(-1) / (2 * grid)
        fy00 = (f01 - f0_1)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fy01 = (f02 - f00)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fy10 = (f11 - f1_1)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fy11 = (f12 - f10)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fx00 = (f10 - f_10)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fx01 = (f20 - f00)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fx10 = (f11 - f_11)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fx11 = (f21 - f01)[self.nindri, self.nindr].reshape(
            self.nat, self.nat, -1).squeeze(-1) / (self.nx1 - self.nx_1)
        fxy00, fxy11 = fy00 * fx00, fx11 * fy11
        fxy01, fxy10 = fx01 * fy01, fx10 * fy10
        return fy00, fy01, fy10, fy11, fx00, fx01, fx10, fx11, fxy00, fxy01, \
            fxy10, fxy11


class Spline1d:
    """Polynomial natural (linear, cubic) non-periodic spline.

    Arguments:
        x: 1D Tensor variable.
        y: 1D (single) or 2D (batch) Tensor variable.

    Keyword Args:
        kind: string, define spline method, default is cubic
        abcd: a tuple of Tensor for cubic spline

    Examples:
        >>> import tbmalt.common.maths.interpolator as interp
        >>> import torch
        >>> import matplotlib.pyplot as plt
        >>> from scipy.interpolate import CubicSpline
        >>> x = torch.linspace(1, 10, 10)
        >>> y = torch.rand(10)
        >>> fit = interp.Spline1d(x, y)
        >>> for ii in torch.linspace(1.1, 9.9, 89):
        >>>     pred = fit(ii)
        >>>     plt.plot(ii, pred, 'r.')
        >>> plt.plot(x, y, 'o', label='grid point')
        >>> plt.legend()
        >>> plt.show()

    References:
        .. https://en.wikipedia.org/wiki/Spline_(mathematics)

    """

    def __init__(self, x: Tensor, y: Tensor, **kwargs):
        self.xp, self.yp = x, y
        kind = kwargs.get('kind')

        if kind in ('cubic', None):
            self.kind = 'cubic'
            if kwargs.get('abcd') is not None:
                self.aa, self.bb, self.cc, self.dd = kwargs.get('abcd')
            else:
                self.aa, self.bb, self.cc, self.dd = self._get_abcd()
        elif kind == 'linear':
            self.kind = 'linear'
        else:
            raise NotImplementedError('%s not implemented' % self.kind)

    def __call__(self, xnew: Tensor):
        """Evaluate the polynomial linear or cubic spline.

        Arguments:
            xnew: 0D Tensor

        Returns:
            ynew: 0D Tensor
        """
        # according to the order to choose spline method
        self.xnew = xnew if xnew.dim() == 1 else xnew.unsqueeze(0)
        self.batch = False if len(self.xnew) == 1 else True
        self.knot = [ii in self.xp for ii in self.xnew]

        # boundary condition of xnew,  xp[0] < xnew < xp[-1]
        assert self.xnew.ge(self.xp[0]).all() and self.xnew.le(self.xp[-1]).all()

        # get the nearest grid point index of d in x
        self.dind = [
            bisect.bisect(self.xp.detach().numpy(), ii.detach().numpy()) - 1
            for ii in self.xnew]
        # generate new index if self.batch
        self.ind = [torch.arange(len(self.xnew)), self.dind] if self.batch else self.dind

        if self.kind == 'cubic':
            return self._cubic()
        elif self.kind == 'linear':
            return self._linear()

    def _linear(self):
        """Calculate linear interpolation."""
        return self.yp[self.ind] + (self.xnew - self.xp[self.dind]) / (
            self.xp[1:] - self.xp[:-1])[self.dind] * (
            self.yp[..., 1:] - self.yp[..., :-1])[self.ind]

    def _cubic(self):
        """Calculate cubic spline interpolation."""
        # calculate a, b, c, d parameters, need input x and y
        dx = self.xnew - self.xp[self.dind]
        return self.aa[self.ind] + self.bb[self.ind] * dx + \
            self.cc[self.ind] * dx ** 2 + self.dd[self.ind] * dx ** 3

    def _get_abcd(self):
        """Get parameter aa, bb, cc, dd for cubic spline interpolation."""
        # get the first dim of x
        nx = self.xp.shape[0]
        assert nx > 3  # the length of x variable must > 3

        # get the differnce between grid points
        dxp = self.xp[1:] - self.xp[:-1]
        dyp = self.yp[..., 1:] - self.yp[..., :-1]

        # get b, c, d from reference website: step 3~9, first calculate c
        A = torch.zeros(nx, nx)
        A.diagonal()[1:-1] = 2 * (dxp[:-1] + dxp[1:])  # diag
        A[torch.arange(nx - 1), torch.arange(nx - 1) + 1] = dxp  # off-diag
        A[torch.arange(nx - 1) + 1, torch.arange(nx - 1)] = dxp
        A[0, 0], A[-1, -1] = 1., 1.
        A[0, 1], A[1, 0] = 0., 0.   # natural condition
        A[-1, -2], A[-2, -1] = 0., 0.

        B = torch.zeros(*self.yp.shape)
        B[..., 1:-1] = 3 * (dyp[..., 1:] / dxp[1:] - dyp[..., :-1] / dxp[:-1])
        B = B.permute(1, 0) if B.dim() == 2 else B

        cc, _ = torch.lstsq(B, A)
        cc = cc.permute(1, 0) if cc.squeeze().dim() == 2 else cc.squeeze()
        bb = dyp / dxp - dxp * (cc[..., 1:] + 2 * cc[..., :-1]) / 3
        dd = (cc[..., 1:] - cc[..., :-1]) / (3 * dxp)
        return self.yp, bb, cc, dd


class SKInterpolation:
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
