"""Parameter continuation with given function
"""
from collections.abc import Callable
from typing import TypeVar, TypeAlias, Any
import numpy
from ._type import is_type_of

Y = TypeVar('Y', numpy.ndarray, float)
P: TypeAlias = dict[str, Any]

def continuation(
    fun: Callable,
    y0: Y,
    params: P,
    end_val: float,
    param_idx: str,
    resolution: int = 100,
    show_progress: bool = False,
) -> list[dict[str, Y | P]]:
    """Parameter continuation

    Parameters
    ----------
    fun : Callable
        Function to call in each step.
    y0 : numpy.ndarray | float
        Value of `y` passed to fun in the initial step.
    params : numpy.ndarray | float
        Value of the system parameters in the initial step.
    end_val : float
        End value of the continuation.
    param_idx : int, optional
        Index of the parameter to continue in `params`.
    resolution : int, optional
        Resolution of the continuation from the current value to `end_val`, by default `100`.

    Returns
    -------
    tuple of y and params
        Final value of `y` and `params`.
    """

    h = (end_val-params[param_idx])/(resolution-1)

    y = y0 if isinstance(y0, float) else y0.copy()
    p = params.copy()

    found: list[dict[str, Y | P]] = []

    for i in range(resolution):
        ret = fun(y, p)

        if not isinstance(ret, ContinuationFunResult):
            raise TypeError

        if not ret.success:
            break

        y = ret.y
        p = ret.p

        if y is None or p is None:
            break

        if not isinstance(y, numpy.ndarray) or y.size == 1:
            y = float(y)

        if not is_type_of(y, type(y0)):
            raise TypeError(type(y), type(y0))
        if not is_type_of(p, type(params)):
            raise TypeError(type(p), type(params))

        if show_progress:
            precision = 10
            show_str: list[str] = ["\tSUCCESS" if ret.success else "FAILURE", f"{i+1:04d}"]
            for val in [y, list(p.values())]:
                if isinstance(val, float) or len(val) == 1:
                    show_str.append(f"{val:+.{precision}f}")
                else:
                    show_str.append(" ".join(["{:+."+str(precision)+"f}"] * len(val)).format(*val))
            print(" ".join(show_str), end="\r")

        found.append({
            'y': y,
            'params': p
        })

        if i != resolution-1:
            p[param_idx] += h
    if show_progress: print()
    return found

class ContinuationFunResult:
    def __init__(self,
        success: bool, y: Y | None, p: P | None,
    ) -> None:
        self.success = success
        self.y = y
        self.p = p
