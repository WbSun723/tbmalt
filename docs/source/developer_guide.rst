*****************
Development Guide
*****************

To ensure the continued health of the TBMaLT package, all python based contributions must
follow the project's style guide. This guide not only ensures that the code is written in
a consistent and above all readable manner, but that the resulting package remains stable,
trivially extensible, and bloat-free. This document is not intended to be a comprehensive
guide but rather an extension of the Google python style guide. Developers should therefore
consult the Google python style guide's documentation for technical details and for information
on the topics not covered here. [*]_ This guide is divided into three main sections covering
docstring formatting, coding-style, and general best practices.


Docstrings
==========
Documentation is integral to helping users understand not only what a function does, but
how it is to be used. Therefore, descriptive and well written docstrings must be provided,
in English, for all functions, classes, generates, etc., including private methods. All
text must be wrapped at the 79'th column; i.e. docstring lines may not exceed a width of
80 characters, including indentation and newline characters, except when giving URLs,
path-names, etc. [*]_ In general docstrings are composed of a one-line-summary followed by
a longer, more detailed summary and then by one or more additional sections.


Sections
--------
While there are many possible sections that can be included, those which are considered
to be the most important are outlined below. In subsequent sections the term ":code:`function`"
(fixed width font) is used to refer to an actual python function instance, whereas "function"
(standard font) is used to mean :code:`functions`, :code:`classes`, :code:`generators`,
etc. in general.


One Line Summary
^^^^^^^^^^^^^^^^
This is a short one-line description that allows users to quickly discern a function's
purpose. This should immediately follow the opening quotes, terminate with a period and be
followed by a blank line, as demonstrated in line-2 & 3 of :ref:`docstring_general`.
One line summaries are mandatory for all functions without exception. While it is not always
possible to convey the full intent of a function in a single line, an attempt should still
be made to do so.


Detailed Summary
^^^^^^^^^^^^^^^^
Following the one line summary, a second more detailed description should be given. This
description must be able to completely describe what a function does and how it is invoked.
[*]_ Explanations of *how* a function works can be left to the :code:`notes` section. Detailed
summaries are generally considered mandatory, however special exceptions may be made, i.e.
for :code:`getters` and :code:`setters`. This section may contain images, tables, paragraphs,
references, maths, examples and so on. A simple example of a :code:`detailed_summary` is
provide in lines 4 to 5 of :ref:`docstring_general`.



Arguments
^^^^^^^^^
Any function that accepts arguments, other than :code:`self`, must contain an ":code:`arguments`"
section in its docstring. This section, whose start is signified by an ":code:`Arguments:`"
header, lists each argument by name, explicitly states its expected type, and provides
an appropriate description. The name and type-info must appear on a single, colon-
terminated line with the description placed on the following, indented, line(s), as
demonstrated in :ref:`docstring_args_2` & :ref:`docstring_general`.

.. code-block:: python
    :caption: Code-block: `Argument` declaration
    :name: docstring_args_1
    :linenos:

    def func(name):
        """...
        Arguments:
            name (type_info):
                A description of the argument should go here. If a multi-line
                description is needed then it should be wrapped like so. Note the
                indentation used.
        ...
        """
        ...



Type info, located within the parenthesis, must always be given, no exceptions are permitted. [*]_
If an argument is type agnostic then the its type should be listed as ":code:`Any`". If an argument
is optional then :code:`None` should be included as a possible type, unless it defaults to a value
other than :code:`None`, in which case :code:`optional` should be included. See
:ref:`docstring_args_2`.

.. code-block:: python
    :caption: Code-block: Type declaration examples
    :name: docstring_args_2
    :linenos:

    def example(a, b, c, d, e, f=None):
        """...
        Arguments:
            a (int):
                An integer.
            b (int or float):
                An integer or a float.
            c (list[int or float]):
                A list of integers and/or floats.
            d (dict[str of Any]):
                A dictionary keyed by strings and valued by any type.
            e (torch.Tensor[Any]):
                A torch tensor with flexible dtyping.
            f (int or None):
                An optional integer. [DEFAULT=None]
        ...
        """
        ...

If there is a default option, then it should be explicitly stated as ":code:`[DEFAULT=val]`"
where :code:`val` is replaced by the argument's default value. Note, this should *not* be
broken over multiple lines. The arguments section of a function's docstring should not only
recount the positional and keyword arguments but also any ":code:`*args`" arguments it
consumes. ":code:`**kwargs`", on the other hand, are documented in their own section.


Keyword Arguments
"""""""""""""""""
Any and all :code:`**kwargs` arguments that are used or consumed by a function should be
documented identically to the standard arguments, albeit in a separate "Keyword Arguments"
section. If :code:`*args` and :code:`**kwargs` are directly passed on to another function
or a parent class then they need not be documented. [*]_


Returns and Yields
^^^^^^^^^^^^^^^^^^
Gives a description for, and specifies the type of, the entity(s) that are returned/yielded
from a function. If returning a single variable, the first line *must* specify the returned
variable's type and name like so :code:`(variable_type): variable_name:`. This is then
followed by a restructured-text style bullet point which gives a description of said
variable; as shown in :ref:`docstring_return_1`.

.. code-block:: python
    :caption: Code-block: `Returns` section (single variable)
    :name: docstring_return_1
    :linenos:

    def example_function():
        """...
        Returns:
            (float): some_returned_variable:
                * A description of ``some_returned_variable`` should then be given
                  here. This format is used to maintain visual consistency with
                  the layout use when returning multiple variables


            If needed a more in-depth discussion can then be given here. As such,
            the "Returns" section should be treated more like a "Notes" section
            than an "Arguments" section.
        ...
        """
        ...

The hanging lines of multi-line variable descriptions must be indented by **exactly two**
spaces with respect to the ":code:`*`" character, any other indentation will break Sphinx.
Optionally, a more in-depth free-form discussion can be given after this, if deemed
necessary. Unlike numpy style returns, Google does not have inbuilt support for elegantly
documenting and parsing multiple returns. The currently accepted workaround for this is to
gather multiple returns into a single tuple. This is permitted as python :code:`functions`
technically return multiple variables as a tuple. Hence, ":code:`(tuple):`" must be appended
to the first line of the returns section as follows:

.. code-block:: python
    :caption: Code-block: `Returns` section (multiple variables)
    :name: docstring_return_2
    :linenos:

    def example_function():
        """...
        Returns:
            (tuple):
            (float): variable_one:
                * A description of ``variable_one`` should be given here.
            (float): variable_two:
                * A description of ``variable_two`` should be given here.

            If needed a more in-depth discussion can then be given here. As such,
            the "Returns" section should be treated more like a "Notes" section
            than an "Arguments" section.
        ...
        """
        ...

Attributes
^^^^^^^^^^
The public attributes of a class should be documented in an :code:`Attributes` section.
This section follows the :code:`Arguments` section(s) and should be documented in an
identical manner. This section is only required when documenting classes with public
attributes.

Notes
^^^^^
In general any additional comments about a function or its usage which do not fit into
any other section can be placed into the :code:`Notes` section. If the function's operation
is complex enough to require a dedicated walk-through, then it should be given here. Any
works on which a function is based, papers, books, etc. should also be mentioned and
referenced in this section.

Raises
^^^^^^
Any exceptions that are manually raised by a function should be documented in the
:code:`Raises` section. This is particularly important when raising custom exceptions.
This section should not only document what exceptions may be raised during operation, but
also the circumstances under which they are raised. :ref:`docstring_raises`
shows how such sections should be formatted.

.. code-block:: python
    :caption: Code-block: `Raises` section
    :name: docstring_raises
    :linenos:

    def example_function(val_1, val_2):
        """...
        Raises:
            AttributeError: Each error that is manually raised should be listed in
                the ``Raises`` section, and a description given specifying under
                what circumstances it is raised.
            ValueError: If `val_1` and `val_2` are equal.
        ...
        """
        ...

Warnings
^^^^^^^^
Any general warning about when a function may fail or where it might do something that the
user may consider unexpected (*gotchas*) should be documented in the free-form :code:`Warnings`
section.

Examples
^^^^^^^^
This section can be used to provide users with examples that illustrate a function's usage.
This should only be used to supplement a function's operational description, not replace
it. The inclusion of an :code:`Examples` section is highly encouraged, but is not mandatory.
The example code given in this section must follow the doctest_ format and should be fully
self-contained. That is to say, the user should be able to copy, paste and run the code
result without modification. Multiple examples should be separated by blank lines,
comments explaining the examples should also have blank lines above and below them.
:ref:`docstring_examples` demonstrates how the :code:`Examples` section is to be
documented.

.. code-block:: python
    :caption: Code-block: `Examples` section
    :name: docstring_examples
    :linenos:

    def example_function(val_1, val_2):
        """...
        Examples:
            Comment explaining the first example.

            >>> from example_module import example_function
            >>> a = 10
            >>> b = 20
            >>> c = example_function(a, b)
            >>> print(c)
            200

            Comment explaining the second example.

            >>> from example_module import example_function
            >>> print(15.5 , 19.28)
            307.21

        ...
        """
        ...


References
^^^^^^^^^^
Any citations made in the notes section should be listed in the :code:`References` section
and must follow the Harvard style. It is expected that comments within a function's code
will also make use of these references. An example of how a reference is made is provided
in code-block :ref:`docstring_references`.

.. code-block:: python
    :caption: Code-block: `References` section
    :name: docstring_references
    :linenos:

    def example_function():
        """...
        Notes:
            A reference is cited like so.[1]_ It must then have a corresponding
            entry in the ``References`` section.

        References:
            .. [1] Hourahine, B. et al. (2020) DFTB+, "a software package for
               efficient approximate density functional theory based atomistic
               simulations", The Journal of chemical physics, 152(12), p. 124101.
        ...
        """
        ...


Putting it all Together
-----------------------

.. code-block:: python
    :caption: Code-block: full docstring example
    :name: docstring_general
    :linenos:

    def example_function(a, b, print_results=False):
        """Calculate the product of two numbers ``a`` & ``b``.

        This function takes two arguments ``a`` & ``b``. Multiplies them together
        & returns the resulting number.

        Arguments:
            a (float or int):
                The first argument, this is a value that is to be multiplied
                against ``b``.
            b (float or int):
                The second argument, this will be multiplied against ``a``.
            print_results (bool, optional):
                If True, the result will be printed prior to being returned.
                [DEFAULT=False]

        Returns:
            (float): c:
                The product of ``a`` & ``b``.

        Notes:
            This is an example of a Python docstring. [1]_ This example is
            overly verbose by intent.

        Examples:
            An example of an example:

            >>> from example import example_function()
            >>> example_function(10, 20)
            200

        Raises:
            ValueError: If ``a`` & ``b`` are equal.

        Warnings:
            This example has never been tested. There is a 1 in 10 chance of this
            code deciding to terminate itself.

        References:
            .. [1] Van Rossum, G. & Drake Jr, F.L., 1995. "Python reference manual,
               Centrum voor Wiskunde en Informatica Amsterdam".

        """
        if a == b:  # <-- Raise an exception if a & b are equal.
            raise ValueError("Can't multiply two equal numbers together.")

        if np.random.rand() < 0.1:  # <-- Pointless exit roulette.
            exit()

        # Calculate the product
        c = a * b

        if print_results:  # <-- Print the result if instructed to do so.
            print(c)

        return c  # <-- Finally return the result


Module Docstrings
-----------------



Miscellaneous
-------------
Docstrings may include UTF-8 characters and images where appropriate; with images saved to
the :code:`doc/images` directory. Additional sections may be included at the developers
desecration. However, while :code:`Todo` section usage is encouraged in development branches
its use in the main branch should generally be avoided. If including maths in the docstring
it is advisable to precede the triple-quote with an :code:`r` to indicate a raw string. This
avoids having to escape every backslash. Docstrings should be parsed by autodoc and visually
inspected prior to submitting a pull request.




Code
====

Comments
--------
Although similar to docstrings, comments should be written to aid other developers rather
than the end user. It is important that comments are detailed enough to allow a new developer
to jump-in at any part of the code and quickly understand exactly what is going on. To this
end, a one-line -- one-comment policy is enforce; i.e. a comment should be provided for
each line of code explaining its action, even those whose purpose may appear obvious. Any
none-standard programming choices should also be justified, e.g. why an obscure set of
instructions was used. Some lines of code may go uncommented if they are detailed in the
previous comment (comment stacking). While this rule is not rigorously enforced, code that
abuses comment stacking may be rejected. Comments are subject to the same column width
restrictions as docstrings, i.e. 80 characters including the new-line and indentation
characters, some exceptions are permitted if they improves readability. Comments can include
UTF-8 characters and cite references in the docstring if needed. Code that follows a
mathematical procedure from a paper or book should include the relevant equations in the
comments to clarify what is being done in a step by step manner. Any deviations from the
reference source should also be clearly stated and justified.



Structure
---------
Code should be written in a manner that ensures modularity, shape-agnosticism, and a
plug-n-play nature. Within the context of this project "shape agnosticism" refers to the
ability of a function to operate on inputs regardless of whether such inputs represent a
single instance or a batch of instances. Shape agnosticism should be applied not only to
the function as a whole but each line of code within it, i.e. a function is not considered
shape agnostic if it contains a :code:`if batch: do A, else: do B` statement. Modularity
refers to the ability to separate the code into independent components which contain only
that which is necessary to their core functionality. Modularity ensures code extensibility
and is conducive to a plug-n-play codebase and supports the ability to take a class or
function and replace it with another, similar one, without requiring additional changes to
the code to be made, i.e. swapping one mixer method for another or being able to drop in one
representation method for another.


Miscellaneous
-------------
In general, coding style should follow the guidelines laid out in the Google style guild.
However, certain points which are considered important are outlined here. Variable names
should be underscore separated and as descriptive as possible, however, commonly accepted
notation is preferred when applicable. For example; a pair of variables holding the
Hamiltonian and Fock matrices should be named :code:`H` and :code:`F` respectively. When
using commonly accepted notation, any violations of PEP8's naming conventions will be waived,
e.g. using a single upper case character as a variable name. When raising exceptions, built-
in exception types should be use wherever possible. However, custom exceptions are permitted
where appropriate. Generally custom exception should be defined in the :code:`common/exceptions.py`
module, and should inherit from the base :code:`TBMaLT` exception or its derivatives. Note
that catch-all excepts are not permitted under any circumstances. Use commonly available
python packages where available and when appropriate, don't try and reinvent the wheel.
This will be enforced to prevent unnecessary code bloat and improve maintainability. All
internal code must be written in a manner consistent with the use of atomic units.


Testing
-------
Every python module in the TBMaLT package must have a corresponding unit-test file
associated with it, named ":code:`test_<name>.py`. These files, located in the
:code:`tests/unittests` directory, must be able to test each component of their associated
modules using the :code:`pytest` package. Wherever possible, best efforts should be made
to isolate the component being tested, as this aids the ability to track down the source
of an error. Unit-tests should verify that functions perform as intended, produce results
within acceptable tolerances, are stable during back-propagation, raise the correct
exceptions in response to erroneous inputs, are GPU operable, batch operability, etc. These
tests should not require any external software or data to be installed or downloaded in order
to run.

In addition to the standard unit-tests there also exist a series of deep tests, located
in the :code:`tests/deeptests` directory. These tests are entirely optional and are
traditionally reserved for testing core functionality. Unlike unit-test these may require
additional data to be downloaded and new software packages, such as DFTB+, to be installed
in order to run.

While tests are expected to provide a high degree of coverage, it is unreasonable to strive
for 100% coverage.



References
==========

Footnotes
---------
.. [*] https://google.github.io/styleguide/pyguide.html
.. [*] See the Google style definition for more information.
.. [*] In conjunction with the arguments and returns section of the docstring.
.. [*] Types must still be specified in the docstring even when using PEP484_.
.. [*] Exception: If the downstream function is private then the arguments should be specified.


Citations
---------

.. _PEP484: https://www.python.org/dev/peps/pep-0484/
.. _doctest: https://docs.python.org/3/library/doctest.html



