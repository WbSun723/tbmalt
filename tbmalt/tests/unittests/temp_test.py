import sys
from typing import Optional, Any, Tuple, List
import torch
from time import time
Tensor = torch.Tensor
#sys.path.append('/home/mcsloy')
from tbmalt.common import maths

def pack(tensors: List[Tensor], axis: int = 0, value: Any = 0,
         size: Optional[Tuple[int]] = None) -> Tensor:
    """Pad and pack a sequence of tensors together.

    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Arguments:
        tensors: List of tensors to be packed, all with identical dtypes.
        axis: Axis along which tensors will be packed; 0 for first axis
            -1 for last axis, etc. This will be a new dimension. [DEFAULT=0]
        value: The value with which the tensor is to be padded. [DEFAULT=0]
        size: Specifies the size to which tensors should be padded. By default
            tensors are padded to the size of the largest tensor. However,
            ``max_size`` can be used to overwrite this behaviour.

    Returns:
        packed_tensors: Input tensors padded and packed into a single tensor.

    Notes:
        ``packed_tensors`` maintains the same order as ``tensors``. This
        is faster & more flexable than the internal pytorch p6ck & pad
        functions (at this particuarl task).

    """

    # If "size" unspecified; the maximum observed size along each axis is used
    if size is None:
        size = torch.max(torch.tensor([i.shape for i in tensors]), 0)[0]

    # Create a tensor to pack into & fill with padding value. Work under the
    # assumption that "axis" == 0 and permute later on (easier this way).
    padded = torch.empty(len(tensors), *size,
                         dtype=tensors[0].dtype,
                         device=tensors[0].device).fill_(value)

    # Loop over tensors & the dimension of "padded" it is to be packed into.
    # This loop is not an elegant solution, but it is fast.
    for n, t in enumerate(tensors):
        # Pack via slice operations to allow code to be dimension agnostic.
        slices = (slice(0, s) for s in t.shape)
        padded[(n, *slices)] = t

    # If "axis" was anything other than 0, then padded must be permuted
    if axis != 0:
        # Resolve negative "axis" values to their 'positive' equivalents to
        # maintain expected slicing behaviour when using the insert function.
        if axis < 0:
            axis = padded.dim() + 1 + axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        ax = list(range(1, padded.dim()))

        # Re-insert the concatenation axis as specified
        ax.insert(axis, 0)

        # Perform the permeation
        padded = padded.permute(ax)

    # Return the packed tensor
    return padded


def test_eighb_general_batch():
    """eighb accuracy on a batch of general eigenvalue problems."""
    torch.manual_seed(0)
    sizes = torch.randint(5, 30, (50,))
    a = [maths.sym(torch.rand(s, s)) for s in sizes]
    b = [maths.sym(torch.eye(s) * torch.rand(s)) for s in sizes]
    a_batch, b_batch = pack(a), pack(b)

    aux_settings = [True, False]
    schemes = ['chol', 'lowd']
    t1 = time()
    for scheme in schemes:
        for aux in aux_settings:
            w_calc = maths.eighb(a_batch, b_batch, scheme=scheme, aux=aux)[0]
    t2 = time()
    return t2 - t1


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.FloatTensor)
    t_cpu = test_eighb_general_batch()
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # t_gpu = test_eighb_general_batch()
    # print(f'CPU: {t_cpu:6.2f}')
    # print(f'GPU: {t_gpu:6.2f}')
    # print(f'GPU/CPU: {t_gpu/t_cpu:6.2f}')
