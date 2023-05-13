from typing import Any, Type, TypeVar, TypeGuard, TypeAlias
from numpy import ndarray

Y = TypeVar("Y", ndarray, float, int, list[float], tuple[float], list[int], tuple[int])
YF = TypeVar(
    "YF", ndarray, float, int, list[float], tuple[float], list[int], tuple[int]
)
YB = TypeVar("YB", float, int)
YC = TypeVar("YC", ndarray, float, tuple)
P: TypeAlias = dict[str, Any]

_T = TypeVar("_T")


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
        ``True`` if target is the given type, and ``False`` otherwise.
    """
    return isinstance(target, type)
