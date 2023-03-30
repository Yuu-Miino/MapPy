class BasicResult:
    """Basic Result class

    The class provides a ``__str__`` method to show public attributes not starting with ``__``.
    It also implements ``__repr__`` as a method to show ``__dict__`` as a string.
    """
    def __init__(self) -> None:
        pass
    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return str({key: val for key, val in self.__dict__.items() if not key.startswith("__")})

