"""Performs tests on functions in the tbmalt.common.maths.interpolator."""
import torch
from torch.autograd import gradcheck
import numpy as np
import pytest
from scipy import linalg
from tbmalt.common import maths, batch
from tbmalt.tests.test_utils import *
from tbmalt.common.maths.interpolator import *
from scipy.interpolate import interp1d, interp2d


##################################
# TBMaLT.common.maths.PolySpline #
##################################
@fix_seed
def test_polyspline(device):
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    y = torch.rand(10, device=device)
    fit = PolySpline(x, y)
    f_ref = interp1d(x, y)
    pred = fit(0.6)
    ref = f_ref(0.6)
    assert  pred - ref < 1E-10

@pytest.mark.grad
@fix_seed
def test_polyspline_grad(device):
    """Gradient evaluation of maths.sym function."""
    x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    y = torch.rand(10, device=device)
    fit = PolySpline(x, y)
    xi = torch.tensor([0.6], requires_grad=True)
    print("gradcheck(PolySpline, (x, y), raise_exception=False)",
          gradcheck(fit, xi, raise_exception=False))
    grad_is_safe = gradcheck(fit, xi, raise_exception=False)

    assert grad_is_safe, 'Gradient stability test'

##################################
# TBMaLT.common.maths.BicubInterpVec #
##################################
def test_bicubic_grid(device):
    x = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
                      [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]])
    z = torch.rand(2, 2, 10, 10, device=device)
    interp = BicubInterpVec(x, z)
    ref = torch.zeros(2, 2)
    for ii in range(2):
        for jj in range(2):
            f = interp2d(x[ii], x[jj], z[ii, jj].squeeze())
            ref[ii, jj] = torch.tensor(f(5., 5.))
    pred = interp(torch.tensor([5., 5.]), torch.tensor([5., 5.]))
    assert  torch.allclose(pred, ref)
