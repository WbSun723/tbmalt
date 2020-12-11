import bisect
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from numbers import Real
Tensor = torch.Tensor



class BicubInterpVec:
    """Vectorized bicubic interpolation method for DFTB.
 
    Arguments:
        xmesh: a 1-D Tensor
        ymesh: a 1-D Tensor
        xmesh: a 1-D Tensor
        ymesh: a 1-D Tensor

    Reference:
        .. https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self, xmesh, zmesh):
        """Get interpolation with two variables."""
        self.xmesh = xmesh
        self.zmesh = zmesh

    def __call__(self, xi, yi):
        """Calculate bicubic interpolation.

        Arguments:
            ix, iy: the interpolation point
        Returns:
            p(x, y) = [1, x, x**2, x**3] * a_mat * [1, y, y**2, y**3].T
        """
        self.xi = xi
        self.yi = yi
        # directly give the coefficients matrices
        coeff = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                          [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = torch.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                           [0., 1., -2., 1.], [0., 0., -1., 1.]])

        # get the nearest grid points indices of xi and yi
        xmesh_ = self.xmesh.cpu() if self.xmesh.device.type == 'cuda' else self.xmesh
        xi_ = self.xi.cpu() if self.xi.device.type == 'cuda' else self.xi
        self.nx0 = [bisect.bisect(xmesh_[ii].detach().numpy(),
                                    xi_[ii].detach().numpy()) - 1
                    for ii in range(len(self.xi))]
        # get all surrounding 4 grid points indices, _1 means previous grid point index
        self.nind = [ii for ii in range(len(self.xi))]
        self.nx1 = [ii + 1 for ii in self.nx0]
        self.nx_1 = [ii - 1 if ii >= 1 else ii for ii in self.nx0]
        self.nx2 = [ii + 2 if ii <= len(self.xmesh) - 3 else ii + 1 for ii in self.nx0]

        # this is to transfer x or y to fraction, with natom element
        x_ = (self.xi - self.xmesh.T[self.nx0, self.nind]) / (self.xmesh.T[
            self.nx1, self.nind] - self.xmesh.T[self.nx0, self.nind])

        # build [1, x, x**2, x**3] matrices of all atoms, dimension: [4, natom]
        xmat = torch.stack([x_ ** 0, x_ ** 1, x_ ** 2, x_ ** 3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self.fmat0th(self.zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = \
            self.fmat1th(self.zmesh, f00, f10, f01, f11)
        fmat = torch.stack([torch.stack([f00, f01, f02, f03]),
                            torch.stack([f10, f11, f12, f13]),
                            torch.stack([f20, f21, f22, f23]),
                            torch.stack([f30, f31, f32, f33])])

        # method 1 to calculate a_mat, not stable
        # a_mat = t.einsum('ii,ijlmn,jj->ijlmn', coeff, fmat, coeff_)
        # return t.einsum('ij,iijkn,ik->jkn', xmat, a_mat, xmat)
        if fmat.dim() == 4:
            a_mat = torch.matmul(torch.matmul(
                coeff, fmat.permute(2, 3, 0, 1)), coeff_)
        elif fmat.dim() == 5:
            a_mat = torch.matmul(torch.matmul(
                coeff, fmat.permute(2, 3, 4, 0, 1)), coeff_)
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


class PolySpline():
    """Polynomial natural (linear, cubic) spline.

    Arguments:
        x: a 1-D Tensor
        y: a 1-D Tensor
        parameter (list, optional): a list of parameters get from grid points,
            e.g., for cubic, parameter=[a, b, c, d]

    Keyword Args:
        kind: string, define spline method
        abcd: a tuple of Tensor

    References:
        .. https://en.wikipedia.org/wiki/Spline_(mathematics)

    """

    def __init__(self, x: Tensor, y: Tensor, **kwargs):
        """Initialize the interpolation class."""
        self.xp = x
        self.yp = y
        kind = kwargs.get('kind')
        if kind in ('cubic', None):
            self.kind = 'cubic'
        if self.kind == 'cubic':
            if kwargs.get('abcd') is not None:
                self.aa, self.bb, self.cc, self.dd = kwargs.get('abcd')
            else:
                self.aa, self.bb, self.cc, self.dd = self.get_abcd()

    def __call__(self, xnew: Union[Tensor, Real]):
        """Evaluate the polynomial spline.

        Arguments:
            xnew: 0-d Tensor or real number

        Returns:
            ynew: 0-d Tensor
        """
        # according to the order to choose spline method
        self.xnew = xnew

        # boundary condition of d
        if not self.xp[0] <= self.xnew <= self.xp[-1]:
            raise ValueError("%s is out of boundary" % self.xnew)

        # get the nearest grid point index of d in x
        if torch.is_tensor(self.xp):
            self.dind = bisect.bisect(self.xp.numpy(), self.xnew) - 1
        elif type(self.xp) is np.ndarray:
            self.dind = bisect.bisect(self.xp, self.xnew) - 1

        if self.kind == 'cubic':
            return self.cubic()

    def linear(self):
        """Calculate linear interpolation."""
        pass

    def cubic(self):
        """Calculate cubic spline interpolation."""
        # calculate a, b, c, d parameters, need input x and y
        dx = self.xnew - self.xp[self.dind]
        return self.aa[self.dind] + self.bb[self.dind] * dx + \
            self.cc[self.dind] * dx ** 2.0 + self.dd[self.dind] * dx ** 3.0

    def get_abcd(self):
        """Get parameter a, b, c, d for cubic spline interpolation."""
        assert self.xp is not None and self.yp is not None

        # get the first dim of x
        self.nx = self.xp.shape[0]

        # get the differnce between grid points
        self.diff_xp = self.xp[1:] - self.xp[:-1]

        # get b, c, d from reference website: step 3~9
        if self.yp.dim() == 1:
            b = torch.zeros(self.nx - 1)
            d = torch.zeros(self.nx - 1)
            A = self.cala()
        else:
            b = torch.zeros(self.nx - 1, self.yp.shape[1])
            d = torch.zeros(self.nx - 1, self.yp.shape[1])

        A = self.cala()
        B = self.calb()

        # a is grid point values
        a = self.yp

        # return c (Ac=B) with least squares and least norm problems
        c, _ = torch.lstsq(B, A)
        for i in range(self.nx - 1):
            b[i] = (a[i + 1] - a[i]) / self.diff_xp[i] - \
                self.diff_xp[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
            d[i] = (c[i + 1] - c[i]) / (3.0 * self.diff_xp[i])
        return a, b, c.squeeze(), d

    def cala(self):
        """Calculate a para in spline interpolation."""
        aa = torch.zeros(self.nx, self.nx)
        aa[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                aa[i + 1, i + 1] = 2.0 * (self.diff_xp[i] + self.diff_xp[i + 1])
            aa[i + 1, i] = self.diff_xp[i]
            aa[i, i + 1] = self.diff_xp[i]

        aa[0, 1] = 0.0
        aa[self.nx - 1, self.nx - 2] = 0.0
        aa[self.nx - 1, self.nx - 1] = 1.0
        return aa

    def calb(self):
        """Calculate b para in spline interpolation."""
        bb = torch.zeros(*self.yp.shape)
        for i in range(self.nx - 2):
            bb[i + 1] = 3.0 * (self.yp[i + 2] - self.yp[i + 1]) / \
                self.diff_xp[i + 1] - 3.0 * (self.yp[i + 1] - self.yp[i]) / \
                self.diff_xp[i]
        return bb



class Bspline():
    """Bspline interpolation for DFTB.

    originate from
    """

    def __init__(self):
        """Revised from the following.

        https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/
        scipy.interpolate.BSpline.html
        """
        pass

    def bspline(self, t, c, k, x):
        n = len(t) - k - 1
        assert (n >= k + 1) and (len(c) >= n)
        return sum(c[i] * self.B(t, k, x, i) for i in range(n))

    def B(self, t, k, x, i):
        if k == 0:
            return 1.0 if t[i] <= x < t[i+1] else 0.0
        if t[i+k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i]) / (t[i+k] - t[i]) * self.B(t, k-1, x, i)
            if t[i + k + 1] == t[i + 1]:
                c2 = 0.0
            else:
                c2 = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * \
                    self.B(t, k - 1, x, i + 1)
        return c1 + c2

    def test(self):
        """Test function for Bspline."""
        tarr = [0, 1, 2, 3, 4, 5, 6]
        carr = [0, -2, -1.5, -1]
        fig, ax = plt.subplots()
        xx = np.linspace(1.5, 4.5, 50)
        ax.plot(xx, [self.bspline(tarr, carr, 2, ix) for ix in xx], 'r-',
                lw=3, label='naive')
        ax.grid(True)
        ax.legend(loc='best')
        plt.show()




