"""This file contains seven functions that differ only in documentation style.
This is intended to test the ability of an IDE to perform type-checking on,
and construct documentation from, various documentation styles. While each
function returns an integer the documentation (incorrectly) states that it will
return a list. This is done to prevent an IDE "cheating" by using type inference.
The type hinting methods are:
    - test_1: In-line PEP 484.
    - test_2: Comment style PEP 484.
    - test_3: Google style docstring.
    - test_4: Hybrid Google style docstring.
    - test_5: Hybrid Google style docstring with PEP 484 return.
    - test_6: Numpy style docstring.
    - test_7: reStructuredTex style docstring.


PyCharm:
    +--------+------+---------+----------+
    |Test No | Docs | Type-in | Type-out |
    +========+======+=========+==========+
    |1       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |2       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |3       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |4       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |5       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |6       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |7       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+


SublimeText 3:
    +--------+------+---------+----------+
    |Test No | Docs | Type-in | Type-out |
    +========+======+=========+==========+
    |1       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |2       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |3       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |4       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |5       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |6       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+
    |7       |  ?   |    ?    |    ?     |
    +--------+------+---------+----------+

Things to check:
    - Docstrings fully displayed
    - Input types defined
    - Output types defined
    - Type checking for inputs
    - Type checking for outputs


"""

def test_1(a: int, b: int) -> list:
    """Test of in-line PEP 484 type hinting.

    Arguments:
        a: Description of arg 1
        b: Description of arg 2

    Returns:
        c: Description of return
    """
    c = a * b
    return c


def test_2(a, b):
    # type: (int, int) -> list
    """Test of backwards compatible comment style PEP 484 type hinting.

    Arguments:
        a: Description of arg 1
        b: Description of arg 2

    Returns:
        c: Description of return
    """
    c = a * b
    return c


def test_3(a, b):
    """Test of pure Google style docstring type hinting.

    Arguments:
        a (int): Description of arg 1
        b (int): Description of arg 2

    Returns:
        list: c: Description of return

    """
    c = a * b
    return c


def test_4(a, b):
    """Test of hybrid Google style docstring type hinting.

    Arguments:
        a (int): Description of arg 1
        b (int): Description of arg 2

    Returns:
        c (list): Description of return

    """
    c = a * b
    return c


def test_5(a, b) -> list:
    """Test of hybrid Google style docstring type hinting with PEP484.

    Arguments:
        a (int): Description of arg 1
        b (int): Description of arg 2

    Returns:
        c (list): Description of return

    """
    c = a * b
    return c


def test_6(a, b):
    """Test of numpy style docstring type hinting.

    Parameters
    ----------
    a : int
        Description of arg 1
    b : int
        Description of arg 2

    Returns
    -------
    c : list
        Description of return

    """
    c = a * b
    return c


def test_7(a, b):
    """Test of numpy style docstring type hinting.


    :param a: Description of arg 1
    :param b: Description of arg 2
    :return: Description of return
    :type a: int
    :type b: int
    :rtype: list

    """
    c = a * b
    return c





import torch
from numbers import Number, Real
from typing import Union, List, Optional, Dict, Any, Literal
Tensor = torch.Tensor


def func_test(i: Optional[Literal['a', 'b', 'c']]) -> int:
    """This is a test.

    Arguments:
        i: some input available options are

            * 'a' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            * 'b' : Trust Region Reflective algorithm, particularly suitable
              for large sparse problems with bounds. Generally robust method.
            - "chol": Cholesky factorisation.


    """
    return 10


def example(a: int, b: Union[int, str], c: List[Real], d: Dict[str, Any],
            e: Tensor, f: Literal['option_1', 'option_2'] = 'option_1',
            g: Optional[int] = None) -> Tensor:
    """...
    Arguments:
        a: an integer.
        b: An integer or a string.
        c: A list of anything numerical and real; integers, floats, etc.
        d: A dictionary keyed by strings and valued by any type.
        e: A torch tensor.
        f: String argument (with default) that can be of of the following:

            - "option_1": the first possible option
            - "option_2": the section option.

            [DEFAULT='option_1']
        g: An optional integer. [DEFAULT=None]

    Returns:
        h: A tensor
    ...
    """
    ...



o = func_test('a')

a_string = 'str'
z_1 = test_1(a_string, a_string)  # <- Should warn you that inputs are incorrect type

an_integer = 10
z_2 = test_4(an_integer, an_integer)  # <- Should not raise any warnings
z_2.append(10)  # <- Should not raise any warnings
_ = z_2 ** 2    # <- Should raise warning about incorrect type
v = test_4(1, 2)
