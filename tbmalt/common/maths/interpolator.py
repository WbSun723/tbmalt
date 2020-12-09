import bisect
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import json
import logging
import argparse
from cvxopt import matrix, solvers
from scipy.linalg import block_diag


class _Interpolate(object):
    """Common features for interpolation.

    Deal with the input data dtype, shape, dimension problems.

    Methods:
        __call__
        set_dtype
        set_shape
    """

    def __init__(self, x=None, y=None, d=None):
        self.set_dtype(x, y)
        self.set_shape(x, y)

    def __call__(self, x):
        pass

    def set_dtype(self, x, y):
        """Check the input dtype."""
        if type(x) is np.ndarray:
            x = t.from_numpy(x)
        if type(y) is np.ndarray:
            x = t.from_numpy(x)

        # return torch type
        x_dtype = x.dtype
        y_dtype = x.dtype

        # x and y dtype should be the same
        if x_dtype != y_dtype:
            raise ValueError("input x, y dtype not the same")
        self.dtype = x_dtype

    def set_shape(self, x, y):
        """Set the shape of x, y."""
        x_shape = x.shape
        if y is not None:
            y_shape = y.shape
            if x_shape == y_shape:
                self.is_same_shape = True
            else:
                self.is_same_shape = False

    def bound(self):
        pass


class BicubInterpVec:
    """Vectorized bicubic interpolation method for DFTB.

    reference: https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self, para, ml):
        """Get interpolation with two variables."""
        self.para = para
        self.ml = ml

    def bicubic_2d(self, xmesh, zmesh, xi, yi):
        """Build fmat.

        [[f(0, 0),  f(0, 1),   f_y(0, 0),  f_y(0, 1)],
         [f(1, 0),   f(1, 1),   f_y(1, 0),  f_y(1, 1)],
         [f_x(0, 0), f_x(0, 1), f_xy(0, 0), f_xy(0, 1)],
         [f_x(1, 0), f_x(1, 1), f_xy(1, 0), f_xy(1, 1)]]
        a_mat = coeff * famt * coeff_

        Args:
            xmesh: x (2D), [natom, ngrid_point]
            zmesh: z (5D), [natom, natom, ngrid_point, ngrid_point, 20]
            ix, iy: the interpolation point
        Returns:
            p(x, y) = [1, x, x**2, x**3] * a_mat * [1, y, y**2, y**3].T
        """
        # check if xi, yi is out of range of xmesh, ymesh
        # xmin = t.ge(xi, self.ml['compressionRMin'])
        # xmax = t.le(xi, self.ml['compressionRMax'])
        # ymin = t.ge(yi, self.ml['compressionRMin'])
        # ymax = t.le(yi, self.ml['compressionRMax'])
        # assert False not in xmin
        # assert False not in xmax
        # assert False not in ymin
        # assert False not in ymax

        # directly give the coefficients matrices
        coeff = t.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.],
                          [-3., 3., -2., -1.], [2., -2., 1., 1.]])
        coeff_ = t.tensor([[1., 0., -3., 2.], [0., 0., 3., -2.],
                           [0., 1., -2., 1.], [0., 0., -1., 1.]])

        # get the nearest grid points indices of xi and yi
        xmesh_ = xmesh.cpu() if xmesh.device.type == 'cuda' else xmesh
        xi_ = xi.cpu() if xi.device.type == 'cuda' else xi
        self.nx0 = [bisect.bisect(xmesh_[ii].detach().numpy(),
                                    xi_[ii].detach().numpy()) - 1
                    for ii in range(len(xi))]
        # get all surrounding 4 grid points indices, _1 means previous grid point index
        self.nind = [ii for ii in range(len(xi))]
        self.nx1 = [ii + 1 for ii in self.nx0]
        self.nx_1 = [ii - 1 if ii >= 1 else ii for ii in self.nx0]
        self.nx2 = [ii + 2 if ii <= len(xmesh) - 3 else ii + 1 for ii in self.nx0]

        # this is to transfer x or y to fraction, with natom element
        x_ = (xi - xmesh.T[self.nx0, self.nind]) / (xmesh.T[
            self.nx1, self.nind] - xmesh.T[self.nx0, self.nind])

        # build [1, x, x**2, x**3] matrices of all atoms, dimension: [4, natom]
        xmat = t.stack([x_ ** 0, x_ ** 1, x_ ** 2, x_ ** 3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self.fmat0th(zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = \
            self.fmat1th(xmesh, zmesh, f00, f10, f01, f11)
        fmat = t.stack([t.stack([f00, f01, f02, f03]),
                        t.stack([f10, f11, f12, f13]),
                        t.stack([f20, f21, f22, f23]),
                        t.stack([f30, f31, f32, f33])])

        # method 1 to calculate a_mat, not stable
        # a_mat = t.einsum('ii,ijlmn,jj->ijlmn', coeff, fmat, coeff_)
        # return t.einsum('ij,iijkn,ik->jkn', xmat, a_mat, xmat)
        a_mat = t.matmul(t.matmul(coeff, fmat.permute(2, 3, 4, 0, 1)), coeff_)
        return t.stack([t.stack(
            [t.matmul(t.matmul(xmat[:, i], a_mat[i, j]), xmat[:, j])
             for j in range(len(xi))]) for i in range(len(xi))])

    def fmat0th(self, zmesh):
        """Construct f(0/1, 0/1) in fmat."""
        f00 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx0[j]]
                                for j in self.nind]) for i in self.nind])
        f10 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx0[j]]
                                for j in self.nind]) for i in self.nind])
        f01 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx1[j]]
                                for j in self.nind]) for i in self.nind])
        f11 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx1[j]]
                                for j in self.nind]) for i in self.nind])
        return f00, f10, f01, f11

    def fmat1th(self, xmesh, zmesh, f00, f10, f01, f11):
        """Get the 1st derivative of four grid points over x, y and xy."""
        f_10 = t.stack([t.stack([zmesh[i, j, self.nx_1[i], self.nx0[j]]
                                 for j in self.nind]) for i in self.nind])
        f_11 = t.stack([t.stack([zmesh[i, j, self.nx_1[i], self.nx1[j]]
                                 for j in self.nind]) for i in self.nind])
        f0_1 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx_1[j]]
                                 for j in self.nind]) for i in self.nind])
        f02 = t.stack([t.stack([zmesh[i, j, self.nx0[i], self.nx2[j]]
                                for j in self.nind]) for i in self.nind])
        f1_1 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx_1[j]]
                                 for j in self.nind]) for i in self.nind])
        f12 = t.stack([t.stack([zmesh[i, j, self.nx1[i], self.nx2[j]]
                                for j in self.nind]) for i in self.nind])
        f20 = t.stack([t.stack([zmesh[i, j, self.nx2[i], self.nx0[j]]
                                for j in self.nind]) for i in self.nind])
        f21 = t.stack([t.stack([zmesh[i, j, self.nx2[i], self.nx1[j]]
                                for j in self.nind]) for i in self.nind])

        # calculate the derivative: (F(1) - F(-1) / (2 * grid)
        # if there is no previous or next grdi point, it will be:
        # (F(1) - F(0) / grid or (F(0) - F(-1) / grid
        fy00 = t.stack([t.stack([(f01[i, j] - f0_1[i, j]) /
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fy01 = t.stack([t.stack([(f02[i, j] - f00[i, j]) /
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fy10 = t.stack([t.stack([(f11[i, j] - f1_1[i, j]) /
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fy11 = t.stack([t.stack([(f12[i, j] - f10[i, j]) /
                                 (self.nx1[j] - self.nx_1[j])
                                 for j in self.nind]) for i in self.nind])
        fx00 = t.stack([t.stack([(f10[i, j] - f_10[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx01 = t.stack([t.stack([(f20[i, j] - f00[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx10 = t.stack([t.stack([(f11[i, j] - f_11[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fx11 = t.stack([t.stack([(f21[i, j] - f01[i, j]) /
                                 (self.nx1[i] - self.nx_1[i])
                                 for j in self.nind]) for i in self.nind])
        fxy00, fxy11 = fy00 * fx00, fx11 * fy11
        fxy01, fxy10 = fx01 * fy01, fx10 * fy10
        return fy00, fy01, fy10, fy11, fx00, fx01, fx10, fx11, fxy00, fxy01, \
            fxy10, fxy11


class PolySpline(_Interpolate):
    """Polynomial natural (linear, cubic) spline.

    See: https://en.wikipedia.org/wiki/Spline_(mathematics)
    You can either generate spline parameters(abcd) or offer spline
    parameters, and get spline interpolation results.

    Args:
        xp (tensor, optional): one dimension grid points
        yp (tensor, optional): one or two dimension grid points
        parameter (list, optional): a list of parameters get from grid points,
            e.g., for cubic, parameter=[a, b, c, d]

    Returns:
        result (tensor): spline interpolation value at dd
    """

    def __init__(self, x=None, y=None, abcd=None, kind='cubic'):
        """Initialize the interpolation class."""
        _Interpolate.__init__(self, x, y)
        self.xp = x
        self.yp = y
        self.abcd = abcd
        self.kind = kind

        # delete original input
        del x, y, abcd, kind

    def __call__(self, dd=None):
        """Evaluate the polynomial spline.

        Args:
        dd : torch.tensor
            Points to evaluate the interpolant at.

        Returns:
        ynew : torch.tensor
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.
        """
        # according to the order to choose spline method
        self.dd = dd

        # if d is not None, it will return spline interpolation values
        if self.dd is not None:

            # boundary condition of d
            if not self.xp[0] <= self.dd <= self.xp[-1]:
                raise ValueError("%s is out of boundary" % self.dd)

            # get the nearest grid point index of d in x
            if t.is_tensor(self.xp):
                self.dind = bisect.bisect(self.xp.numpy(), self.dd) - 1
            elif type(self.xp) is np.ndarray:
                self.dind = bisect.bisect(self.xp, self.dd) - 1

        if self.kind == 'linear':
            self.ynew = self.linear()
        elif self.kind == 'cubic':
            self.ynew = self.cubic()
        else:
            raise NotImplementedError("%s is unsupported" % self.kind)
        return self.ynew

    def linear(self):
        """Calculate linear interpolation."""
        pass

    def cubic(self):
        """Calculate cubic spline interpolation."""
        # calculate a, b, c, d parameters, need input x and y
        if self.abcd is None:
            a, b, c, d = self.get_abcd()
        else:
            a, b, c, d = self.abcd

        dx = self.dd - self.xp[self.dind]
        return a[self.dind] + b[self.dind] * dx + c[self.dind] * dx ** 2.0 + d[self.dind] * dx ** 3.0

    def get_abcd(self):
        """Get parameter a, b, c, d for cubic spline interpolation."""
        assert self.xp is not None and self.yp is not None

        # get the first dim of x
        self.nx = self.xp.shape[0]

        # get the differnce between grid points
        self.diff_xp = self.xp[1:] - self.xp[:-1]

        # get b, c, d from reference website: step 3~9
        if self.yp.dim() == 1:
            b = t.zeros(self.nx - 1)
            d = t.zeros(self.nx - 1)
            A = self.cala()
        else:
            b = t.zeros(self.nx - 1, self.yp.shape[1])
            d = t.zeros(self.nx - 1, self.yp.shape[1])

        A = self.cala()
        B = self.calb()

        # a is grid point values
        a = self.yp

        # return c (Ac=B) with least squares and least norm problems
        c, _ = t.lstsq(B, A)
        for i in range(self.nx - 1):
            b[i] = (a[i + 1] - a[i]) / self.diff_xp[i] - \
                self.diff_xp[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
            d[i] = (c[i + 1] - c[i]) / (3.0 * self.diff_xp[i])
        return a, b, c.squeeze(), d

    def _get_abcd(self):
        """Get parameter a, b, c, d for cubic spline interpolation."""
        # get the first dim of x
        self.nx = self.xp.shape[0]

        # get the differnce between grid points
        self.h = self.xp[1:] - self.xp[:-1]

        # get the differnce between grid points
        self.ydiff = self.yp[1:] - self.yp[:-1]

        # setp 6, define l, mu, z
        ll = t.zeros(self.nx, dtype=self.dtype)
        mu = t.zeros(self.nx, dtype=self.dtype)
        zz = t.zeros(self.nx, dtype=self.dtype)
        alpha = t.zeros(self.nx, dtype=self.dtype)
        ll[0] = ll[-1] = 1.

        # step 7, calculate alpha, l, mu, z
        for i in range(1, self.nx - 1):
            alpha[i] = 3. * self.ydiff[i] / self.h[i] - \
                3. * self.ydiff[i - 1] / self.h[i - 1]
            ll[i] = 2 * (self.xp[i + 1] - self.xp[i - 1]) - \
                self.h[i - 1] * mu[i - 1]
            mu[i] = self.h[i] / ll[i]
            zz[i] = (alpha[i] - self.h[i - 1] * zz[i - 1]) / ll[i]

        # step 8, define b, c, d
        b = t.zeros(self.nx, dtype=self.dtype)
        c = t.zeros(self.nx, dtype=self.dtype)
        d = t.zeros(self.nx, dtype=self.dtype)

        # step 9, get b, c, d
        for i in range(self.nx - 2, -1, -1):
            c[i] = zz[i] - mu[i] * c[i + 1]
            b[i] = self.ydiff[i] / self.h[i] - \
                self.h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / 3 / self.h[i]

        return self.yp, b, c, d

    def cala(self):
        """Calculate a para in spline interpolation."""
        aa = t.zeros(self.nx, self.nx)
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
        bb = t.zeros(*self.yp.shape, dtype=self.diff_xp.dtype)
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


class Spline1:
    def __init__(self, x, y, d):
        self.xp = x
        self.yp = y
        self.dd = d
        self.nx = self.xp.shape[0]
        self.h = self.xp[1:] - self.xp[:-1]
        self.diffy = self.yp[1:] - self.yp[:-1]
        self.dind = bisect.bisect(self.xp.numpy(), self.dd) - 1
        self.ynew = self.cubic()

    def cubic(self):
        A = self.get_A()
        B = self.get_B()
        a = t.zeros(self.nx)
        c = t.zeros(self.nx)
        d = t.zeros(self.nx)

        # least squares and least norm problems
        M, _ = t.lstsq(B, A)
        for i in range(self.nx - 2):
            # b[i] = (self.xp[i + 1] * M[i]- self.xp[i] * M[i + 1]) / self.diff_xp[i] / 2
            c[i + 1] = (self.diffy[i]) / self.h[i + 1] - self.h[i + 1] / 6 * (M[i + 1] - M[i])
            d[i + 1] = (self.xp[i + 1] * self.yp[i] - self.xp[i] * self.yp[i + 1]) / self.h[i + 1] - \
            self.h[i + 1] / 6 * (self.xp[i + 1] * M[i] - self.xp[i] * M[i + 1])

        return (M[self.dind] * (self.xp[self.dind + 1] - self.dd) ** 3 + \
                M[self.dind + 1] * (self.dd - self.xp[self.dind]) ** 3) / self.xp[self.dind] / 6 + \
            c[self.dind + 1] * self.dd + d[self.dind + 1]

    def get_B(self):
        # natural boundary condition, the first and last are zero
        B = t.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 6.0 * ((self.diffy[i + 1]) / self.h[i + 1] -
                              (self.diffy[i]) / self.h[i]) / \
            (self.h[i + 1] + self.h[i])
        return B

    def get_A(self):
        """Calculate a para in spline interpolation."""
        A = t.zeros(self.nx, self.nx)
        A[0, 0] = 1.
        # A[0, 1] = 1.
        for i in range(self.nx - 2):
            A[i + 1, i + 1] = 2.
            A[i + 1, i] = self.h[i] / (self.h[i + 1] + self.h[i])
            A[i + 1, i + 2] = 1 - A[i + 1, i]
        A[self.nx - 1, self.nx - 1] = 1.
        # A[self.nx - 1, self.nx - 2] = 1.
        return A


