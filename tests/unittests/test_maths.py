"""Performs tests on functions present in the tbmalt.common.maths module"""
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

from tbmalt.common import maths
import numpy as np
import pytest
import warnings
import torch
torch.autograd.set_detect_anomaly(True)

# Make sure the code does not get committed to the main TBMaLT branch before
# the import path is done a more appropriate manner. Once this is done os, sys
# and warnings will no longer need to be imported.
# TODO: Fix module import for maths_test.py; this will require the correct
#  directory structure to be implemented first.
warnings.warn('tbmalt.common.maths must be imported correctly')

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
    run_on_gpu(test_gaussian_single)


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
        for i in range(n)
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
    """Batch evaluation test."""
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
    run_on_gpu(test_hellinger_single)
