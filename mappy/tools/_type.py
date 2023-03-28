from typing import Any, Type, TypeVar, TypeGuard

_T = TypeVar('_T')

def is_type_of(target: Any, type: Type[_T]) -> TypeGuard[_T]:
    return isinstance(target, type)
