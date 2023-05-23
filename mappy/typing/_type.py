from typing import Any, Type, TypeVar, TypeGuard, TypeAlias
from numpy import ndarray

Y = TypeVar("Y", ndarray, float, int, list[float], tuple[float], list[int], tuple[int])
YF = TypeVar(
    "YF", ndarray, float, int, list[float], tuple[float], list[int], tuple[int]
)
YB = TypeVar("YB", float, int)
YC = TypeVar(
    "YC", ndarray, float, int, list[float], tuple[float], list[int], tuple[int], tuple
)
P: TypeAlias = dict[str, Any]

_T = TypeVar("_T")


def is_type_of(target: Any, type: Type[_T]) -> TypeGuard[_T]:
    """
    Check if the target is of a given type using TypeGuard.

    Parameters
    ----------
    target : Any
        The target to be checked.
    type : Type
        The type to check against.

    Returns
    -------
    TypeGuard[_T]
        Returns ``True`` if the target is of the given type, and ``False`` otherwise.
    """
    return isinstance(target, type)
