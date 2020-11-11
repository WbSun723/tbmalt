"""Performs tests on functions present in the tbmalt.common.maths module"""
import os  # <- temp
import sys  # <- temp
import warnings  # <- temp
import torch
import numpy as np
import pytest
from scipy.linalg import eigh
sys.path.insert(0, os.path.abspath('../../'))
from tbmalt.common import maths

# Default must be set to float64 otherwise gradcheck will not function
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)


def run_on_gpu(func):
    """Runs a function on the GPU if present, skips otherwise."""
    # Skip this test if there is no gpu found.
    if torch.cuda.device_count() == 0:
        pytest.skip('Cannot run GPU test: no GPU found.')
    else:
        # Rerun the test function on the GPU
        with torch.cuda.device(1):
            func()


################################
# TBMaLT.common.maths.gaussian #
################################
def _gaussian_reference(x, mu, sigma):
    """numpy reference method for the gaussian function."""
    return (np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi)))


def test_gaussian_single():
    """Single point evaluation test."""
    # Generate random x, mu & sigma values
    x, mu, sigma = torch.rand(3, dtype=torch.float64)

    # Evaluate the Gaussian function
    pred = maths.gaussian(x, mu, sigma)

    # Evaluate the numpy reference
    ref = _gaussian_reference(x.item(), mu.item(), sigma.item())

    # Calculate the absolute error
    abs_delta = abs(pred.item() - ref)

    # Assert that the errors are within tolerance
    assert abs_delta < 1E-12


def test_gaussian_batch():
    """Batch evaluation test."""
    x, mu, sigma = torch.rand(3, 100, dtype=torch.float64)

    # Evaluate the Gaussian function
    pred = maths.gaussian(x, mu, sigma)

    # Evaluate the numpy reference
    ref = _gaussian_reference(x.numpy(), mu.numpy(), sigma.numpy())

    # Calculate the maximum absolute error
    abs_delta = max(abs(pred.numpy().ravel() - ref.ravel()))

    # Assert that the errors are within tolerance
    assert abs_delta < 1E-12


def test_gaussian_backprop():
    """Back propagation continuity test"""
    # Generate random x, mu & sigma values
    x, mu, sigma = torch.rand(3, dtype=torch.float64, requires_grad=True)

    # Check the gradient
    grad_is_safe = torch.autograd.gradcheck(maths.gaussian, (x, mu, sigma))

    # Assert the gradient stability
    assert grad_is_safe


def test_gaussian_gpu():
    """GPU operability test."""
    run_on_gpu(test_gaussian_batch)


#################################
# TBMaLT.common.maths.hellinger #
#################################
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


def test_hellinger_single():
    """Single point evaluation test."""
    # Generate a pair of random skewed normal distributions
    p, q = _random_skewed_norm(2)

    # Evaluate hellinger distance between the two distributions
    pred = maths.hellinger(torch.tensor(p), torch.tensor(q))

    # Evaluate the numpy reference
    ref = _hellinger_reference(p, q)

    # Calculate the absolute error
    abs_delta = abs(pred.item() - ref)

    # Assert that the errors are within tolerance
    assert abs_delta < 1E-12


def test_hellinger_batch():
    """Batch evaluation test."""
    # Generate some random skewed normal distribution pairs
    p, q = _random_skewed_norm(10), _random_skewed_norm(10)

    # Evaluate hellinger distance between the two distributions
    pred = maths.hellinger(torch.tensor(p), torch.tensor(q))

    # Evaluate the numpy reference
    ref = _hellinger_reference(p, q)

    # Calculate the maximum absolute error
    abs_delta = max(abs(pred.numpy().ravel() - ref.ravel()))

    # Assert that the errors are within tolerance
    assert abs_delta < 1E-12


def test_hellinger_backprop():
    """Back propagation continuity test"""
    # Generate a random skewed normal distribution pair & add 1E-4 to prevent
    # gradcheck from generating negative numbers.
    p, q = torch.tensor(_random_skewed_norm(2) + 1E-4, requires_grad=True,
                        dtype=torch.float64)

    # Check the gradient
    grad_is_safe = torch.autograd.gradcheck(maths.hellinger, (p, q))

    # Assert the gradient stability
    assert grad_is_safe


def test_hellinger_gpu():
    """GPU operability test."""
    run_on_gpu(test_hellinger_batch)


################################
# TBMaLT.common.maths._SymEigB #
################################
def test_symeig_broadened_single():
    """Single point evaluation test."""

    # The forward pass is more of a sanity check as it just makes a call to
    # "torch.symeig".

    # Ensure data is random but consistent
    np.random.seed(1)

    # Generate random matrix to diagonalise
    matrix = np.random.rand(6, 6)
    matrix += matrix.T

    # Get the calculated and reference values
    ref_w, ref_v = eigh(matrix)
    calc_w, calc_v = maths.symeig_broadened(torch.tensor(matrix))

    # Check eigenvalues are comparable
    abs_delta = np.max(np.abs(calc_w.numpy() - ref_w))

    # Assert that the errors are within tolerance
    assert abs_delta < 1E-12

    # Check the resulting eigenvectors are orthogonal
    dots = np.dot(calc_v.numpy(), calc_v.numpy().T)
    np.fill_diagonal(dots, 0.0)  # <- ignore diagonals
    abs_delta_2 = np.max(np.abs(dots))
    assert abs_delta_2 < 1E-12


def test_symeig_broadened_batch():
    """Batch evaluation test."""

    from tbmalt.common import batch

    # This tests the functions ability to deal with "ghost" eigen values and
    # vectors caused by padding matrices during the packing process.

    # Ensure data is random but consistent
    np.random.seed(1)

    # Generate test data
    sizes = torch.randint(2, 10, (10,))  # <- Random sizes
    mats = [torch.rand(s, s) for s in sizes]  # <- random matrices
    mats = [mat + mat.T for mat in mats]  # <- symmetrisation

    # Pack into a single tensor padded with zeros.
    mats_packed = batch.pack(mats)

    # Perform the eigen decomposition calculation.
    w_calc, v_calc = maths.symeig_broadened(mats_packed, sortout=True)

    # Calculate the reference eigenvalues and pack them together. Don't bother
    # with the eigenvectors as they are non-deterministic.
    w_ref = batch.pack([torch.symeig(m)[0] for m in mats])

    # If anything has gone wrong with the "sortout" subroutine or the
    # calculation it will be picked up here.
    abs_delta = torch.max(torch.abs(w_calc - w_ref))

    # Assert that the errors are within tolerance
    assert abs_delta < 1E-12


def test_symeig_broadened_backprop():
    """Back propagation continuity test"""

    # Proxy functions are needed to enforce symmetry
    def inbuilt(m):
        v = (m + m.T) / 2
        return torch.symeig(v, eigenvectors=True)

    def no_broadening(m):
        v = (m + m.T) / 2
        return maths.symeig_broadened(v, method='none')

    def conditional_broadening(m):
        v = (m + m.T) / 2
        return maths.symeig_broadened(v, method='cond')

    def lorentzian_broadening(m):
        v = (m + m.T) / 2
        return maths.symeig_broadened(v, method='lorn')

    np.random.seed(1)

    # Create random matrix, and one with a known degenerate states
    a = torch.rand((10, 10), requires_grad=True) + 10

    # Functions to test: 0) inbuilt method, 1) modified but without broadening,
    # 2) with conditional broadening & 3) with Lorentzian broadening.
    funcs = [inbuilt, no_broadening, conditional_broadening, lorentzian_broadening]
    for func in funcs:
        grad_is_safe = torch.autograd.gradcheck(func, a)
        assert grad_is_safe, f'Non-degenerate test failed on {func.__name__}'

    # torch.autograd.gradcheck seems to fail on degenerate systems or those like
    # torch.tensor([[5., 10.], [10., 5.]]) which don't produce degenerate eigen-
    # valued systems. The main intent of this function is to prevent NaN's from
    # appearing in the code.

    # b = torch.tensor([
    #     [0., 1, 1],
    #     [1, 0., 1],
    #     [1, 1, 0.]],
    #     requires_grad=True)

    # Repeat with known degenerate states; but skip the inbuilt symeig and non-
    # bordering tests as they will always fail.
    # for func in funcs[:2]:
    #     grad_is_safe = torch.autograd.gradcheck(func, b)
    #     assert grad_is_safe, f'Degenerate test failed on {func.__name__}'


def test_symeig_broadened_gpu():
    """GPU operability test."""
    run_on_gpu(test_symeig_broadened_batch)
