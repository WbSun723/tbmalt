import bisect
import torch
from typing import Tuple, Union, Optional
from numbers import Real
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
        .. https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self, xmesh: Tensor, zmesh: Tensor):
        """Get interpolation with two variables."""
        self.xmesh = xmesh
        self.zmesh = zmesh

    def __call__(self, xi: Tensor):
        """Calculate bicubic interpolation.

        Arguments:
            xi: the interpolation points of each atom in molecule

        Notes:
            For each bicubic interpolation, there is 4 surrounding grid points
            and 12 second nearest neighbors. We use _ to represent previous
            points.
        """
        self.xi = xi
        xi_ = self.xi.cpu() if self.xi.device.type == 'cuda' else self.xi
        coeff = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                              [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = torch.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                               [0., 1., -2., 1.], [0., 0., -1., 1.]])

        # get the nearest grid points indices of xi
        self.nx0 = [bisect.bisect(self.xmesh[ii].cpu().detach().numpy(),
                                  xi_[ii].cpu().detach().numpy()) - 1
                    for ii in range(len(self.xi))]

        # get all surrounding 4 grid points indices
        self.nind = [ii for ii in range(len(self.xi))]
        self.nx1 = [ii + 1 for ii in self.nx0]
        self.nx_1 = [ii - 1 if ii >= 1 else ii for ii in self.nx0]
        self.nx2 = [ii + 2 if ii <= len(self.xmesh) - 3 else ii + 1 for ii in self.nx0]

        # this is to transfer x to fraction
        xf = (self.xi - self.xmesh.T[self.nx0, self.nind]) / (self.xmesh.T[
            self.nx1, self.nind] - self.xmesh.T[self.nx0, self.nind])
        xmat = torch.stack([xf ** 0, xf ** 1, xf ** 2, xf ** 3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self.fmat0th(self.zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = \
            self.fmat1th(self.zmesh, f00, f10, f01, f11)
        fmat = torch.stack([torch.stack([f00, f01, f02, f03]),
                            torch.stack([f10, f11, f12, f13]),
                            torch.stack([f20, f21, f22, f23]),
                            torch.stack([f30, f31, f32, f33])])

        pdim = [2, 3, 0, 1] if fmat.dim() == 4 else [2, 3, 4, 0, 1]
        a_mat = torch.matmul(torch.matmul(coeff, fmat.permute(pdim)), coeff_)
        return torch.stack([torch.stack(
            [torch.matmul(torch.matmul(xmat[:, i], a_mat[i, j]), xmat[:, j])
             for j in range(len(xi))]) for i in range(len(xi))])

    def fmat0th(self, zmesh):
        """Construct f(0/1, 0/1) in fmat."""
        f00 = torch.stack([torch.stack([zmesh[i, j, self.nx0[i], self.nx0[j]]
                                        for j in self.nind]) for i in self.nind])
        f10 = torch.stack([torch.stack([zmesh[i, j, self.nx1[i], self.nx0[j]]
                                        for j in self.nind]) for i in self.nind])
        f01 = torch.stack([torch.stack([zmesh[i, j, self.nx0[i], self.nx1[j]]
                                        for j in self.nind]) for i in self.nind])
        f11 = torch.stack([torch.stack([zmesh[i, j, self.nx1[i], self.nx1[j]]
                                        for j in self.nind]) for i in self.nind])
        return f00, f10, f01, f11

    def fmat1th(self, zmesh, f00, f10, f01, f11):
        """Get the 1st derivative of four grid points over x, y and xy."""
        f_10 = torch.stack([torch.stack([zmesh[i, j, self.nx_1[i], self.nx0[j]]
                                         for j in self.nind]) for i in self.nind])
        f_11 = torch.stack([torch.stack([zmesh[i, j, self.nx_1[i], self.nx1[j]]
                                         for j in self.nind]) for i in self.nind])
        f0_1 = torch.stack([torch.stack([zmesh[i, j, self.nx0[i], self.nx_1[j]]
                                         for j in self.nind]) for i in self.nind])
        f02 = torch.stack([torch.stack([zmesh[i, j, self.nx0[i], self.nx2[j]]
                                        for j in self.nind]) for i in self.nind])
        f1_1 = torch.stack([torch.stack([zmesh[i, j, self.nx1[i], self.nx_1[j]]
                                         for j in self.nind]) for i in self.nind])
        f12 = torch.stack([torch.stack([zmesh[i, j, self.nx1[i], self.nx2[j]]
                                        for j in self.nind]) for i in self.nind])
        f20 = torch.stack([torch.stack([zmesh[i, j, self.nx2[i], self.nx0[j]]
                                        for j in self.nind]) for i in self.nind])
        f21 = torch.stack([torch.stack([zmesh[i, j, self.nx2[i], self.nx1[j]]
                                        for j in self.nind]) for i in self.nind])

        # calculate the derivative: (F(1) - F(-1) / (2 * grid)
        # if there is no previous or next grdi point, it will be:
        # (F(1) - F(0) / grid or (F(0) - F(-1) / grid
        fy00 = torch.stack([torch.stack([(f01[i, j] - f0_1[i, j]) /
                                         (self.nx1[j] - self.nx_1[j])
                                         for j in self.nind]) for i in self.nind])
        fy01 = torch.stack([torch.stack([(f02[i, j] - f00[i, j]) /
                                         (self.nx1[j] - self.nx_1[j])
                                         for j in self.nind]) for i in self.nind])
        fy10 = torch.stack([torch.stack([(f11[i, j] - f1_1[i, j]) /
                                         (self.nx1[j] - self.nx_1[j])
                                         for j in self.nind]) for i in self.nind])
        fy11 = torch.stack([torch.stack([(f12[i, j] - f10[i, j]) /
                                         (self.nx1[j] - self.nx_1[j])
                                         for j in self.nind]) for i in self.nind])
        fx00 = torch.stack([torch.stack([(f10[i, j] - f_10[i, j]) /
                                         (self.nx1[i] - self.nx_1[i])
                                         for j in self.nind]) for i in self.nind])
        fx01 = torch.stack([torch.stack([(f20[i, j] - f00[i, j]) /
                                         (self.nx1[i] - self.nx_1[i])
                                         for j in self.nind]) for i in self.nind])
        fx10 = torch.stack([torch.stack([(f11[i, j] - f_11[i, j]) /
                                         (self.nx1[i] - self.nx_1[i])
                                         for j in self.nind]) for i in self.nind])
        fx11 = torch.stack([torch.stack([(f21[i, j] - f01[i, j]) /
                                         (self.nx1[i] - self.nx_1[i])
                                         for j in self.nind]) for i in self.nind])
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
        abcd: a tuple of Tensor

    Methods:
        __call__

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
                self.aa, self.bb, self.cc, self.dd = self.get_abcd()
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
            return self.cubic()
        elif self.kind == 'linear':
            return self.linear()

    def linear(self):
        """Calculate linear interpolation."""
        return self.yp[self.ind] + (self.xnew - self.xp[self.dind]) / (
            self.xp[1:] - self.xp[:-1])[self.dind] * (
            self.yp[..., 1:] - self.yp[..., :-1])[self.ind]

    def cubic(self):
        """Calculate cubic spline interpolation."""
        # calculate a, b, c, d parameters, need input x and y
        dx = self.xnew - self.xp[self.dind]
        return self.aa[self.ind] + self.bb[self.ind] * dx + \
            self.cc[self.ind] * dx ** 2 + self.dd[self.ind] * dx ** 3

    def get_abcd(self):
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
