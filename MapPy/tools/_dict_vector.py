"""Vector accepting dictionary
"""
class DictVector:
    """Class for vectors accepting the dictionary input.

    Example
    -------
    >>> class Parameter(DictVector):
    ...     pass
    >>> data = {'a': 0.2, 'b': 0.1, 'c': -0.3}
    >>> p = Parameter(data)
    >>> p.a
    0.2
    >>> p.b
    0.1
    >>> p.c
    -0.3
    >>> p
    {'a': 0.2, 'b': 0.1, 'c': -0.3}
    """
    def __init__(self, states: dict[str, float] |  None = None) -> None:
        if states is not None:
            for key, val in states.items():
                setattr(self, key, val)
    def __repr__(self) -> str:
        return str(self.__dict__)

