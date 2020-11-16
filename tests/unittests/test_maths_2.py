"""Performs tests on functions present in the tbmalt.common.maths module"""
import os  # <- temp
import sys  # <- temp
import warnings  # <- temp
import torch
import functools
import numpy as np
import pytest
from scipy import linalg
sys.path.insert(0, os.path.abspath('../../'))
from tbmalt.common import maths
from tbmalt.common import batch

# Default must be set to float64 otherwise gradcheck will not function
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


def fix_seed(func):
    """Sets torch's & numpy's random number generator seed.

    Fixing the random number generator's seed maintains consistency between
    tests by ensuring that the same test data is used every time. If this is
    not done it can make debugging problems very difficult.

    Arguments:
        func (function):
            The function that is being wrapped.

    Returns:
        wrapped (function):
            The wrapped function.
    """
    # Use functools.wraps to maintain the original function's docstring
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # Set both numpy's and pytorch's seed to zero
        np.random.seed(0)
        torch.manual_seed(0)

        # Call the function and return its result
        return func(*args, **kwargs)

    # Return the wapped function
    return wrapper


def clean_zero_padding(m, sizes):
    """Removes perturbations induced in the zero padding values by gradcheck.

    When performing gradient stability tests via PyTorch's gradcheck function
    small perturbations are induced in the input data. However, problems are
    encountered when these perturbations occur in the padding values. These
    values should always be zero, and so the test is not truly representative.
    Furthermore, this can even prevent certain tests from running. Thus this
    function serves to remove such perturbations in a gradient safe manner.

    Note that this is intended to operate on 3D matrices where. Specifically a
    batch of square matrices.

    Arguments:
        m (torch.Tensor):
            The tensor whose padding is to be cleaned.
        sizes (torch.Tensor):
            The true sizes of the tensors.

    Returns:
        cleaned (torch.Tensor):
            Cleaned tensor.
    """

    # First identify the maximum tensor size
    max_size = torch.max(sizes)

    # Build a mask that is True anywhere that the tensor should be zero, i.e.
    # True for regions of the tensor that should be zero padded.
    mask_1d = (
            (torch.arange(max_size) - sizes.unsqueeze(1))>= 0
    ).repeat(max_size, 1, 1)

    # This, rather round about, approach to generating and applying the masks
    # must be used as some PyTorch operations like masked_scatter do not seem
    # to function correctly
    mask_full = torch.zeros(*m.shape).bool()
    mask_full[mask_1d.permute(1, 2, 0)] = True
    mask_full[mask_1d.transpose(0, 1)] = True

    # Create and apply the subtraction mask
    temp = torch.zeros_like(m)
    temp[mask_full] = m[mask_full]
    cleaned = m - temp

    return cleaned


def case_eighb


@fix_seed
def test_eighb_standard_single():
    """eighb accuracy on a single standard eigenvalue problem."""
    a = maths.sym(torch.rand(10, 10))

    w_ref = linalg.eigh(a)[0]

    w_calc, v_calc = maths.eighb(a)

    mae_w = torch.max(torch.abs(w_calc - w_ref))
    mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

    assert mae_w < 1E-12, 'Eigenvalue tolerance test'
    assert mae_v < 1E-12, 'Eigenvector orthogonality test'


@fix_seed
def test_eighb_standard_batch():
    """eighb accuracy on a batch of standard eigenvalue problems."""
    sizes = torch.randint(2, 10, (11,))
    a = [maths.sym(torch.rand(s, s)) for s in sizes]
    a_batch = batch.pack(a)

    w_ref = batch.pack([torch.tensor(linalg.eigh(i)[0]) for i in a])

    w_calc = maths.eighb(a_batch)[0]

    mae_w = torch.max(torch.abs(w_calc - w_ref))

    assert mae_w < 1E-12, 'Eigenvalue tolerance test'


@fix_seed
def test_eighb_general_single():
    """eighb accuracy on a single general eigenvalue problem."""
    a = maths.sym(torch.rand(10, 10))
    b = maths.sym(torch.eye(10) * torch.rand(10))

    w_ref = linalg.eigh(a, b)[0]

    schemes = ['cholesky', 'lowdin']
    for scheme in schemes:
        w_calc, v_calc = maths.eighb(a, b, scheme=scheme)

        mae_w = torch.max(torch.abs(w_calc - w_ref))
        mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

        assert mae_w < 1E-12, f'Eigenvalue tolerance test {scheme}'
        assert mae_v < 1E-12, f'Eigenvector orthogonality test {scheme}'


@fix_seed
def test_eighb_general_batch():
    """eighb accuracy on a batch of general eigenvalue problems."""
    sizes = torch.randint(2, 10, (11,))
    a = [maths.sym(torch.rand(s, s)) for s in sizes]
    b = [maths.sym(torch.eye(s) * torch.rand(s)) for s in sizes]
    a_batch, b_batch = batch.pack(a), batch.pack(b)

    w_ref = batch.pack([torch.tensor(linalg.eigh(i, j)[0]) for i, j in zip(a, b)])

    aux_settings = [True, False]
    schemes = ['cholesky', 'lowdin']
    for scheme in schemes:
        for aux in aux_settings:
            w_calc = maths.eighb(a_batch, b_batch, scheme=scheme, aux=aux)[0]

            mae_w = torch.max(torch.abs(w_calc - w_ref))

            assert mae_w < 1E-12, f'Eigenvalue tolerance test {scheme}'


@fix_seed
def test_eighb_broadening_grad():
    """eighb gradient stability on standard, broadened, eigenvalue problems.

    There is no separate test for the standard eigenvalue problem without
    broadening as this would result in a direct call to torch.symeig which is
    unnecessary. However, it is important to note that conditional broadening
    technically is never tested, i.e. the lines:

    .. code-block:: python
        ...
        if ctx.bm == 'cond':  # <- Conditional broadening
            deltas = 1 / torch.where(torch.abs(deltas) > bf,
                                     deltas, bf) * torch.sign(deltas)
        ...

    of `_SymEigB` are never actual run. This is because it only activates when
    there are true eigen-value degeneracies; & degenerate eigenvalue problems
    do not "play well" with the gradcheck operation.
    """

    def eigen_proxy(m, method, size_data=None):
        m = maths.sym(m)
        if size_data is not None:
            m = clean_zero_padding(m, size_data)
        if method is None:
            return torch.symeig(m, True)
        else:
            return maths.eighb(m, broadening_method=method)

    # Generate a single standard eigenvalue test instance
    a1 = maths.sym(torch.rand(10, 10))
    a1.requires_grad = True

    broadening_methods = [None, 'none', 'cond', 'lorn']
    for method in broadening_methods:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a1, method))
        assert grad_is_safe, f'Non-degenerate single test failed on {method}'

    # Generate a batch of standard eigenvalue test instances
    sizes = torch.randint(4, 10, (12,))
    a2 = batch.pack([maths.sym(torch.rand(s, s)) for s in sizes])
    a2.requires_grad = True

    for method in broadening_methods[2:]:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a2, method, sizes))
        assert grad_is_safe, f'Non-degenerate batch test failed on {method}'


@fix_seed
def test_eighb_general_grad():
    """eighb gradient stability on general eigenvalue problems."""
    def eigen_proxy(m, n, scheme, size_data=None):
        m, n = maths.sym(m), maths.sym(n)
        if size_data is not None:
            m = clean_zero_padding(m, size_data)
            n = clean_zero_padding(n, size_data)

        return maths.eighb(m, n, scheme=scheme)

    # Generate a single generalised eigenvalue test instance
    a1 = maths.sym(torch.rand(10, 10))
    b1 = maths.sym(torch.eye(10) * torch.rand(10))
    a1.requires_grad = True
    b1.requires_grad = True

    schemes = ['cholesky', 'lowdin']
    for scheme in schemes:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a1, b1, scheme))
        assert grad_is_safe, f'Non-degenerate single test failed on {scheme}'

    # Generate a batch of generalised eigenvalue test instances
    sizes = torch.randint(4, 10, (12,))
    a2 = batch.pack([maths.sym(torch.rand(s, s)) for s in sizes])
    b2 = batch.pack([maths.sym(torch.eye(s) * torch.rand(s)) for s in sizes])
    a2.requires_grad, b2.requires_grad = True, True

    for scheme in schemes:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a2, b2, scheme, sizes))
        assert grad_is_safe, f'Non-degenerate batch test failed on {scheme}'

