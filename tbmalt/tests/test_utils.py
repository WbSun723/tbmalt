# -*- coding: utf-8 -*-
r"""This is a simple example of a module level docstring.

All modules should be provided with a short docstring detailing its function.
It does not need to list the classes and methods present only public module-
level variables, like so:

Attributes:
    some_level_variable (int): Each module-level variable should be described.

A further freeform (or structured) discussion can be given if deemed necessary.
Note that the docstring is immediately preceded by a short line specifying the
documents encoding ``# -*- coding: utf-8 -*-``.
"""

import numpy as np
import torch
import functools
import pytest
from contextlib import contextmanager

# Default must be set to float64 otherwise gradcheck will not function
torch.set_default_dtype(torch.float64)
# This will track for any anomalys in the gradient
torch.autograd.set_detect_anomaly(True)


# _dtype_to_gpu_tensor = {
#     torch.float16: torch.cuda.HalfTensor,
#     torch.float32: torch.cuda.FloatType,
#     torch.float64: torch.cuda.DoubleTensor,
# }
#
# _dtype_to_cpu_tensor = {
#     torch.float16: torch.HalfTensor,
#     torch.float32: torch.FloatType,
#     torch.float64: torch.DoubleTensor,
# }

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


# @contextmanager
# def on_gpu():
#     old_tensor_type = torch.get_default_te
#     _dtype_to_gpu_tensor[torch.get_default_dtype()]
#     torch.set_default_tensor_type()

#torch.cuda.is_available()


#
# class GPURunner:
#
#     def __init__(self):
#         self.default_dtype = torch.get_default_dtype()
#
#     def __enter__(self):
#         torch.set_default_dtype()
#
#     def __exit__(self):

def run_on_gpu(func):
    # GIVE A BETTER DESCRIPTION OF WHAT THIS IS DOING
    """Runs a function on the GPU if present, skips otherwise."""
    # Skip this test if there is no gpu found.
    if torch.cuda.device_count() == 0:
        pytest.skip('Cannot run GPU test: no GPU found.')
    else:
        # Rerun the test function on the GPU and make sure the default
        # dtype is a cuda.
        with torch.cuda.device(1):
            func()

