"""Performs tests on functions present in the tbmalt.common.maths module"""
import torch
import numpy as np
import pytest
from scipy import linalg
from tbmalt.common import maths, batch
from tbmalt.tests.test_utils import *


####################
# Helper Functions #
####################
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
            (torch.arange(max_size) - sizes.unsqueeze(1)) >= 0
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


def _gaussian_reference(x, mu, sigma):
    """numpy reference method for the gaussian function."""
    return (np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi)))


def _hellinger_reference(p, q):
    """numpy reference method for the hellinger function."""
    return np.sqrt(
        np.sum(
            (np.sqrt(p) - np.sqrt(q)) ** 2,
            -1)
    ) / np.sqrt(2)


def _random_skewed_norm(n):
    """Generate a random skewed normal distribution.

    This function will construct & return a specified number of skewed normal
    distributions using scipy's stats.skewnorm function.

    Arguments:
        n (int):
            Number of distributions to return.

    Returns:
        (np.ndarray): distributions:
            * A n by 100 array where each row is a separate distribution.

    """
    from scipy.stats import skewnorm

    # The range of the distributions
    x_values = np.linspace(-6.0, 6.0, 100)

    # Generate an array of `n` distributions
    distributions = np.array([
        skewnorm.pdf(
            # The x values
            x_values,
            # How much to skew the distribution by
            2.5 - (np.random.rand() * 5),
            # How much to scale the distribution by
            scale=np.random.rand() * 3)
        for _ in range(n)
    ])

    # Return the distributions
    return distributions

###########################
# TBMaLT.common.maths.sym #
###########################
@pytest.fixture
def device():
    return torch.device('cpu:0')

@fix_seed
def test_sym_single(device):
    """Serial evaluation of maths.sym function."""
    data = torch.rand(10, 10, device=device)
    pred = maths.sym(data)
    ref = (data + data.T) / 2
    abs_delta = torch.max(torch.abs(pred - ref))
    same_device = pred.device() == device

    assert abs_delta < 1E-12, 'Sanity check'
    assert same_device, 'Check for change of device'


@fix_seed
def test_sym_batch(device):
    """Batch evaluation of maths.sym function."""
    data = torch.rand(10, 10, 10, device=device)
    pred = maths.sym(data, -1, -2)
    ref = torch.stack([(i + i.T) / 2 for i in data], 0)
    abs_delta = torch.max(torch.abs(pred - ref))

    assert abs_delta < 1E-12, 'Sanity check'


@pytest.mark.grad
@fix_seed
def test_sym_grad(device):
    """Gradient evaluation of maths.sym function."""
    data = torch.rand(10, 10, 10, requires_grad=True, device=device)
    grad_is_safe = torch.autograd.gradcheck(maths.sym, (data, -1, -2),
                                            raise_exception=False)

    assert grad_is_safe, 'Gradient stability test'


# @pytest.mark.gpu
# @fix_seed
# def test_sym_gpu(device):
#     """GPU evaluation of maths.sym function."""
#     run_on_gpu(test_sym_batch)


################################
# TBMaLT.common.maths.gaussian #
################################
@fix_seed
def test_gaussian_single(device):
    """Single point evaluation test of Gaussian function."""

    x, mu, sigma = torch.rand(3, device=device)
    pred = maths.gaussian(x, mu, sigma)
    ref = _gaussian_reference(x.item(), mu.item(), sigma.item())
    abs_delta = abs(pred.item() - ref)

    assert abs_delta < 1E-12, 'Tolerance test'


@fix_seed
def test_gaussian_batch(device):
    """Batch evaluation test of Gaussian function."""

    x, mu, sigma = torch.rand(3, 100, 4, device=device)
    pred = maths.gaussian(x, mu, sigma)
    ref = _gaussian_reference(x.numpy(), mu.numpy(), sigma.numpy())
    abs_delta = max(abs(pred.numpy().ravel() - ref.ravel()))

    assert abs_delta < 1E-12, 'Tolerance test'


@pytest.mark.grad
@fix_seed
def test_gaussian_grad(device):
    """Back propagation continuity test of Gaussian function"""

    x, mu, sigma = torch.rand(3, requires_grad=True, device=device)
    grad_is_safe = torch.autograd.gradcheck(maths.gaussian, (x, mu, sigma),
                                            raise_exception=False)

    assert grad_is_safe, 'Gradient stability test'


# @pytest.mark.gpu
# @fix_seed
# def test_gaussian_gpu(device):
#     """GPU operability test of Gaussian function."""
#     run_on_gpu(test_gaussian_batch)


#################################
# TBMaLT.common.maths.hellinger #
#################################


@fix_seed
def test_hellinger_single(device):
    """Single point test of the hellinger distance function."""

    # Generate a pair of random skewed normal distributions
    p, q = _random_skewed_norm(2)

    # Evaluate hellinger distance between the two distributions
    pred = maths.hellinger(torch.tensor(p, device=device),
                           torch.tensor(q, device=device))

    # Evaluate the numpy reference
    ref = _hellinger_reference(p, q)

    # Calculate the absolute error
    abs_delta = abs(pred.item() - ref)

    # Assert that the errors are within tolerance
    assert abs_delta < 1E-12, 'Tolerance test'


@fix_seed
def test_hellinger_batch(device):
    """Batch test of the hellinger distance function."""
    np.random.seed(0)
    p, q = _random_skewed_norm(10), _random_skewed_norm(10)
    pred = maths.hellinger(torch.tensor(p, device=device),
                           torch.tensor(q, device=device))
    ref = _hellinger_reference(p, q)
    abs_delta = max(abs(pred.numpy().ravel() - ref.ravel()))

    assert abs_delta < 1E-12, 'Tolerance test'


@pytest.mark.grad
@fix_seed
def test_hellinger_grad(device):
    """Back propagation continuity test of the hellinger distance function"""

    # Generate a random skewed normal distribution pair & add 1E-4 to prevent
    # gradcheck from generating negative numbers.
    p, q = torch.tensor(_random_skewed_norm(2) + 1E-4, requires_grad=True,
                        device=device)

    # Check the gradient
    grad_is_safe = torch.autograd.gradcheck(maths.hellinger, (p, q),
                                            raise_exception=False)

    # Assert the gradient stability
    assert grad_is_safe, 'Gradient stability test'


# @pytest.mark.gpu
# @fix_seed
# def test_hellinger_gpu(device):
#     """GPU operability test of the hellinger distance function."""
#     run_on_gpu(test_hellinger_batch)


#############################
# TBMaLT.common.maths.eighb #
#############################
@fix_seed
def test_eighb_standard_single(device):
    """eighb accuracy on a single standard eigenvalue problem."""
    a = maths.sym(torch.rand(10, 10, device=device))

    w_ref = linalg.eigh(a)[0]

    w_calc, v_calc = maths.eighb(a)

    mae_w = torch.max(torch.abs(w_calc - w_ref))
    mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

    assert mae_w < 1E-12, 'Eigenvalue tolerance test'
    assert mae_v < 1E-12, 'Eigenvector orthogonality test'


@fix_seed
def test_eighb_standard_batch(device):
    """eighb accuracy on a batch of standard eigenvalue problems."""
    sizes = torch.randint(2, 10, (11,), device=device)
    a = [maths.sym(torch.rand(s, s, device=device)) for s in sizes]
    a_batch = batch.pack(a)

    w_ref = batch.pack([torch.tensor(linalg.eigh(i)[0]) for i in a])

    w_calc = maths.eighb(a_batch)[0]

    mae_w = torch.max(torch.abs(w_calc - w_ref))

    assert mae_w < 1E-12, 'Eigenvalue tolerance test'


@fix_seed
def test_eighb_general_single(device):
    """eighb accuracy on a single general eigenvalue problem."""
    a = maths.sym(torch.rand(10, 10, device=device))
    b = maths.sym(torch.eye(10, device=device)
                  * torch.rand(10, device=device))

    w_ref = linalg.eigh(a, b)[0]

    schemes = ['chol', 'lowd']
    for scheme in schemes:
        w_calc, v_calc = maths.eighb(a, b, scheme=scheme)

        mae_w = torch.max(torch.abs(w_calc - w_ref))
        mae_v = torch.max(torch.abs((v_calc @ v_calc.T).fill_diagonal_(0)))

        assert mae_w < 1E-12, f'Eigenvalue tolerance test {scheme}'
        assert mae_v < 1E-12, f'Eigenvector orthogonality test {scheme}'


@fix_seed
def test_eighb_general_batch(device):
    """eighb accuracy on a batch of general eigenvalue problems."""
    sizes = torch.randint(2, 10, (11,), device=device)
    a = [maths.sym(torch.rand(s, s, device=device)) for s in sizes]
    b = [maths.sym(torch.eye(s, device=device)
                   * torch.rand(s, device=device)) for s in sizes]
    a_batch, b_batch = batch.pack(a), batch.pack(b)

    w_ref = batch.pack([torch.tensor(linalg.eigh(i, j)[0]) for i, j in zip(a, b)])

    aux_settings = [True, False]
    schemes = ['chol', 'lowd']
    for scheme in schemes:
        for aux in aux_settings:
            w_calc = maths.eighb(a_batch, b_batch, scheme=scheme, aux=aux)[0]

            mae_w = torch.max(torch.abs(w_calc - w_ref))

            assert mae_w < 1E-12, f'Eigenvalue tolerance test {scheme}'


@pytest.mark.grad
@fix_seed
def test_eighb_broadening_grad(device):
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

    def eigen_proxy(m, target_method, size_data=None):
        m = maths.sym(m)
        if size_data is not None:
            m = clean_zero_padding(m, size_data)
        if target_method is None:
            return torch.symeig(m, True)
        else:
            return maths.eighb(m, broadening_method=target_method)

    # Generate a single standard eigenvalue test instance
    a1 = maths.sym(torch.rand(8, 8, device=device))
    a1.requires_grad = True

    broadening_methods = [None, 'none', 'cond', 'lorn']
    for method in broadening_methods:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a1, method),
                                                raise_exception=False)
        assert grad_is_safe, f'Non-degenerate single test failed on {method}'

    # Generate a batch of standard eigenvalue test instances
    sizes = torch.randint(3, 8, (5,), device=device)
    a2 = batch.pack([maths.sym(torch.rand(s, s, device=device)) for s in sizes])
    a2.requires_grad = True

    for method in broadening_methods[2:]:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a2, method, sizes),
                                                raise_exception=False)
        assert grad_is_safe, f'Non-degenerate batch test failed on {method}'


@pytest.mark.grad
@fix_seed
def test_eighb_general_grad(device):
    """eighb gradient stability on general eigenvalue problems."""
    def eigen_proxy(m, n, target_scheme, size_data=None):
        m, n = maths.sym(m), maths.sym(n)
        if size_data is not None:
            m = clean_zero_padding(m, size_data)
            n = clean_zero_padding(n, size_data)

        return maths.eighb(m, n, scheme=target_scheme)

    # Generate a single generalised eigenvalue test instance
    a1 = maths.sym(torch.rand(8, 8, device=device))
    b1 = maths.sym(torch.eye(8, device=device) * torch.rand(8, device=device))
    a1.requires_grad = True
    b1.requires_grad = True

    schemes = ['chol', 'lowd']
    for scheme in schemes:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a1, b1, scheme),
                                                raise_exception=False)
        assert grad_is_safe, f'Non-degenerate single test failed on {scheme}'

    # Generate a batch of generalised eigenvalue test instances
    sizes = torch.randint(3, 8, (5,), device=device)
    a2 = batch.pack([maths.sym(torch.rand(s, s, device=device)) for s in sizes])
    b2 = batch.pack([maths.sym(torch.eye(s, device=device) * torch.rand(s, device=device)) for s in sizes])
    a2.requires_grad, b2.requires_grad = True, True

    for scheme in schemes:
        grad_is_safe = torch.autograd.gradcheck(eigen_proxy, (a2, b2, scheme, sizes),
                                                raise_exception=False)
        assert grad_is_safe, f'Non-degenerate batch test failed on {scheme}'

#
# @pytest.mark.gpu
# @fix_seed
# def test_symeig_broadened_gpu(device):
#     """GPU operability test of a batched standard eigenvalue problem."""
#     run_on_gpu(test_eighb_standard_batch)

