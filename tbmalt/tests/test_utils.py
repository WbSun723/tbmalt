import numpy as np
import torch
import functools
import pytest

# Default must be set to float64 otherwise gradcheck will not function
torch.set_default_dtype(torch.float64)
# This will track for any anomalys in the gradient
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


def run_on_gpu(func):
    """Runs a function on the GPU if present, skips otherwise."""
    # Skip this test if there is no gpu found.
    if torch.cuda.device_count() == 0:
        pytest.skip('Cannot run GPU test: no GPU found.')
    else:
        # Rerun the test function on the GPU
        with torch.cuda.device(1):
            func()
