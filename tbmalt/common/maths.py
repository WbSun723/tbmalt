"""A collection of common mathematical functions.

This module contains a collection of back-propagatable mathematical functions.

"""
import torch
import numpy as np


def gaussian(x, mean, std):
    r"""Gaussian distribution function.

    A one dimensional Gaussian function representing the probability density
    function of a normal distribution. This Gaussian takes on the form:

    .. math::

        g(x) = \frac{1}{\sigma\sqrt{2\pi}}e
            \left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\right)

    Where σ (`std`) is the standard deviation, μ (`mean`) is the mean & x is
    the point at which the distribution is to be evaluated. Multiple values
    can be passed for batch operation, see the `Notes` section for further
    information.

    Arguments:
        x (torch.Tensor or float):
            Value(s) at which to evaluate the gaussian function.
        mean (torch.Tensor or float):
            Expectation value(s), i.e the mean.
        std (torch.Tensor or float):
            Standard deviation.

    Returns:
        g (torch.Tensor):
            The gaussian function(s) evaluated at the specified `x`, `mean` &
            `std` value(s).

    Raises:
        TypeError: Raised if neither `x` or `mu` are of type torch.tensor.

    Notes:
        Multiple `x`, `mean` & `std` values can be specified for batch-wise
        evaluation. Note that at least one argument must be a torch.Tensor
        entity, specifically `x` or `mean`, else this function will error out.
        However, zero dimensional tensors are acceptable.

    Examples:
        Evaluating multiple points within a single distribution:

        >>> import tbmalt.common.maths as tb_maths
        >>> x = torch.linspace(0, 1, 100)
        >>> y = tb_maths.gaussian(x, 0.5, 0.5)
        >>> plt.plot(x, y, '-k')
        >>> plt.show()

        Evaluating points on a pair of distributions with differing means:

        >>> x = torch.linspace(0, 1, 100)
        >>> y1, y2 = tb_maths.gaussian(x, torch.tensor([0.25, 0.75]), 0.5)
        >>> plt.plot(x, y1, '-r')
        >>> plt.plot(x, y2, '-b')
        >>> plt.show()

    """
    # Evaluate the gaussian at the specified value(s) and return the result
    return (torch.exp(-0.5 * torch.pow((x - mean) / std, 2))
            / (std * np.sqrt(2 * np.pi)))


def hellinger(p, q):
    r"""Calculate the Hellinger distance between pairs of 1D distributions.

    The Hellinger distance can be used to quantify the similarity between a
    pair of discrete probability distributions which have been evaluated at
    the same sample points.

    Arguments:
        p (torch.Tensor):
            Values observed in the first distribution.
        q (torch.Tensor):
            Values observed in the second distribution.

    Returns:
        distance (torch.Tensor):
            The Hellinger distance between each `p`, `q` distribution pair.

    Notes:
        The Hellinger distance is computed as:

        .. math::

             H(p,q)= \frac{1}{\sqrt{2}}\sqrt{\sum_{i=1}^{k}
                \left( \sqrt{p_i} - \sqrt{q_i}  \right)^2}

        Multiple pairs of distributions can be evaluated simultaneously by
        passing in a 2D torch.Tensor in place of a 1D one.

    Raises:
        ValueError: When elements in `p` or `q` are found to be negative.

    Warnings:
        As `p` and `q` ar probability distributions they must be positive. If
        not, a terminal error will be encountered during backpropagation.

    """
    # Raise a ValueError if negative values are encountered. Negative values
    # will throw an error during backpropagation of the sqrt function.
    if torch.sum(p.detach() < 0) != 0 or torch.sum(q.detach() < 0) != 0:
        raise ValueError('All elements in p & q must be positive.')

    # Calculate & return the Hellinger distance between distribution pair(s)
    # Note that despite what the pytorch documentation states torch.sum does
    # in-fact take the "axis" argument.
    return torch.sqrt(
        torch.sum(
            torch.pow(torch.sqrt(p) - torch.sqrt(q), 2),
            -1)
    ) / np.sqrt(2)