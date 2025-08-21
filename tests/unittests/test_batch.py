"""Performs tests on functions present in the tbmalt.common.batch module"""
import torch
from torch.autograd import gradcheck
import pytest
import numpy as np
from tbmalt.common.batch import pack, unpack, merge, deflate, psort, \
    pargsort, prepeat_interleave
from tests.test_utils import fix_seed


@fix_seed
def test_pack(device):
    """Sanity test of batch packing operation."""
    # Generate matrix list
    sizes = torch.randint(2, 8, (10,))
    matrices = [torch.rand(i, i, device=device) for i in sizes]
    # Pack matrices into a single tensor
    packed = pack(matrices)
    # Construct a numpy equivalent
    max_size = max(packed.shape[1:])
    ref = np.stack(
        np.array([np.pad(i.sft(), (0, max_size-len(i))) for i in matrices]))
    equivalent = np.all((packed.sft() - ref) < 1E-12)
    same_device = packed.device == device

    assert equivalent, 'Check pack method against numpy'
    assert same_device, 'Device persistence check (packed tensor)'

    # Check that the mask is correct
    *_, mask = pack([
        torch.rand(1, device=device),
        torch.rand(2, device=device),
        torch.rand(3, device=device)],
        return_mask=True)

    ref_mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                            dtype=torch.bool, device=device)

    same_device_mask = mask.device == device
    eq = torch.all(torch.eq(mask, ref_mask))

    assert eq, 'Mask yielded an unexpected result'
    assert same_device_mask, 'Device persistence check (mask)'


@pytest.mark.grad
@fix_seed
def test_pack_grad(device):
    """Gradient stability test of batch packing operation."""
    sizes = torch.randint(2, 6, (3,))
    tensors = [torch.rand(i, i, device=device, requires_grad=True) for i in sizes]

    def proxy(*args):
        # Proxy function is used to prevent an undiagnosed error from occurring.
        return pack(list(args))

    grad_is_safe = gradcheck(proxy, tensors, raise_exception=False)
    assert grad_is_safe, 'Gradient stability test'


@fix_seed
def test_sort(device):
    """Ensures that the ``psort`` and ``pargsort`` methods work as intended.

    Notes:
        A separate check is not needed for the ``pargsort`` method as ``psort``
        just wraps around it.
    """

    # Test on with multiple different dimensions
    for d in range(1, 4):
        tensors = [torch.rand((*[i] * d,), device=device) for i in
                   np.random.randint(3, 10, (10,))]

        packed, mask = pack(tensors, return_mask=True)

        pred = psort(packed, mask).values
        ref = pack([i.sort().values for i in tensors])

        check_1 = (pred == ref).all()
        assert check_1, 'Values were incorrectly sorted'

        check_2 = pred.device == device
        assert check_2, 'Device persistence check failed'


def test_deflate(device):
    """General operational test of the `deflate` method."""
    a = torch.tensor([
        [1, 1, 0, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ], device=device)

    # Check 1: single system culling, should remove the last column & row.
    check_1 = (deflate(a) == a[:-1, :-1]).all()
    assert check_1, 'Failed to correctly deflate a single system'

    # Check 2: batch system culling (axis=0), should remove last column.
    check_2 = (deflate(a, axis=0) == a[:, :-1]).all()
    assert check_2, 'Failed to correctly deflate a batch system (axis=0)'

    # Check 3: batch system culling (axis=1), should remove last row.
    check_3 = (deflate(a, axis=1) == a[:-1, :]).all()
    assert check_3, 'Failed to correctly deflate a batch system (axis=1)'

    # Check 4: Check value argument is respected, should do nothing here.
    check_4 = (deflate(a, value=-1) == a).all()
    assert check_4, 'Failed to ignore an unpadded system'

    # Check 5: ValueError should be raised if axis is specified & tensor is 1d.
    with pytest.raises(ValueError, match='Tensor must be at*'):
        deflate(a[0], axis=0)

    # Check 6: high dimensionality tests (warning: this is dependent on `pack`)
    tensors = [torch.full((i, i+1, i+2, i+3, i+4), i, device=device)
               for i in range(10)]
    over_packed = pack(tensors, size=torch.Size((20, 19, 23, 20, 30)))
    check_6 = (deflate(over_packed, axis=0) == pack(tensors)).all()
    assert check_6, 'Failed to correctly deflate a large batch system'

    # Check 7: ensure the result is placed on the correct device
    check_7 = deflate(a).device == device
    assert check_7, 'Result was returned on the wrong device'


@pytest.mark.grad
@fix_seed
def test_deflate_grad(device):
    """Check the gradient stability of the deflate function."""
    def proxy(tensor):
        # Clean the padding values to prevent unjust failures
        proxy_tensor = torch.zeros_like(tensor)
        proxy_tensor[~mask] = tensor[~mask]
        return deflate(proxy_tensor)

    a = torch.tensor([
        [1, 1, 0, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0.],
    ], device=device, requires_grad=True)

    mask = a == 0
    mask = mask.detach()

    check = gradcheck(proxy, a, raise_exception=False)
    assert check, 'Gradient stability check failed'


def test_merge(device):
    """Operational tests of the merge function.

    Warnings:
        This test is depended upon the `pack` function. Thus it will fail if
        the `pack` function is in error.
    """
    # Check 1: ensure the expected result is returned
    a = [torch.full((i, i + 1, i + 2, i + 3, i + 4), float(i), device=device)
         for i in range(6)]

    merged = merge([pack(a[:2]), pack(a[2:4]), pack(a[4:])])
    packed = pack(a)
    check_1 = (merged == packed).all()
    assert check_1, 'Merge attempt failed'

    # Check 2: test axis argument's functionality
    merged = merge([pack(a[:2], axis=1), pack(a[2:4], axis=1),
                    pack(a[4:], axis=1)], axis=1)
    packed = pack(a, axis=1)
    check_2 = (merged == packed).all()
    assert check_2, 'Merge attempt failed when axis != 0'

    # Check 3: device persistence check
    check_3 = packed.device == device
    assert check_3, 'Device persistence check failed'


@pytest.mark.grad
def test_merge_grad(device):
    """Checks gradient stability of the merge function."""
    def proxy(a_in, b_in):
        # Clean padding values
        a_proxy = torch.zeros_like(a_in)
        b_proxy = torch.zeros_like(b_in)
        a_proxy[~a_mask] = a_in[~a_mask]
        b_proxy[~b_mask] = b_in[~b_mask]
        return merge([a_proxy, b_proxy])

    a = torch.tensor([
        [0, 1, 0],
        [2, 3, 4.]],
        device=device, requires_grad=True)

    b = torch.tensor([
        [5, 6, 7, 0],
        [8, 9, 10, 11.]],
        device=device, requires_grad=True)

    a_mask = (a == 0).detach()
    b_mask = (b == 0).detach()

    check = gradcheck(proxy, (a, b), raise_exception=False)
    assert check, 'Gradient stability check failed'


def test_unpack(device):
    """Ensures unpack functions as intended.

    Notes:
        The `unpack` function does not require an in-depth test as it is just
        a wrapper for the `deflate` method. Hence, no grad check exists.

    Warnings:
        This test and the method that is being tested are both dependent on
        the `deflate` method.
    """
    # Check 1: Unpacking without padding
    a = torch.tensor([
        [0, 1, 2, 0],
        [3, 4, 5, 0],
        [0, 0, 1, 0]],
        device=device)

    # Check 1: ensure basic results are correct
    check_1 = all([(i == deflate(j)).all() for i, j in zip(unpack(a), a)])
    assert check_1, 'Failed to unpack'

    # Check 2: ensure axis declaration is obeyed
    check_2 = all([(i == deflate(j)).all() for i, j in
                   zip(unpack(a, axis=1), a.T)])
    assert check_2, 'Failed to respect "axis" declaration'

    # Check 3: device persistence check.
    check_3 = all(i.device == device for i in unpack(a))
    assert check_3, 'Device persistence check failed'


def test_prepeat_interleave(device):
    array = torch.tensor([[1.1, 0], [2.2, 3.3]], device=device)
    repeats = torch.tensor([[1, 0], [2, 3]], device=device)
    repeated = prepeat_interleave(array, repeats, 0)
    reference = torch.tensor([[1.1000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [2.2000, 2.2000, 3.3000, 3.3000, 3.3000]],
                              device=device)

    # Check 1: device persistence check.
    check_1 = repeated.device == device
    assert check_1, 'Device persistence check failed'

    # Check 2: ensure results are correct
    check_2 = torch.all(repeated == reference)
    assert check_2, 'Failed to correctly repeat target tensor'
