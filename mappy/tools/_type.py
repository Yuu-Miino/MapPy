from typing import Any, Type, TypeVar, TypeGuard

_T = TypeVar('_T')

def is_type_of(target: Any, type: Type[_T]) -> TypeGuard[_T]:
    """Check target is a given type with TypeGuard

    Parameters
    ----------
    target : Any
        Target of the function.
    type : Type
        Type to check.

    Returns
    -------
    bool
        True if target is the given type, and False otherwise.
    """
    return isinstance(target, type)
