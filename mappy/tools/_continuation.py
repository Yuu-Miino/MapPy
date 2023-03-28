"""Parameter continuation with given function
"""
from collections.abc import Callable
from typing import TypeVar
import numpy

Y = TypeVar('Y', numpy.ndarray, float)
P = TypeVar('P', numpy.ndarray, float)

__all__ = [
    "continuation"
]

def continuation(
    fun: Callable,
    y0: Y,
    params: P,
    end_val: float,
    param_idx: int = 0,
    resolution: int = 100,
    verbose: bool = True,
    verbose_precision: int = 10,
) -> tuple[Y, P]:
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
    verbose : bool, optional
        Verbose option, by default True
    verbose_precision : int, optional
        Precision of numerics in verbose output, by default `10`.

    Returns
    -------
    list of y and params
        Final value of `y` and `params`.
    """
    if isinstance(params, float):
        h = (end_val-params)/resolution
    else:
        h = (end_val-params[param_idx])/resolution

    y = y0 if isinstance(y0, float) else y0.copy()
    p = params.copy()
    for i in range(resolution+1):
        ret = fun(y, p)

        if hasattr(ret, 'success'):
            if not ret.success:
                break

        if verbose:
            if isinstance(params, float):
                out = f"{p:.{verbose_precision}f}"
            else:
                out = " ".join(["{:."+str(verbose_precision)+"f}"] * len(p)).format(*p)
            if hasattr(ret, "dump_values"):
                out += " "+ret.dump_values(precision=verbose_precision)
            print(out)

        if hasattr(ret, 'y'):
            y = ret.y
        if i != resolution:
            if isinstance(params, float):
                p += h
            else:
                p[param_idx] += h
    return (y, p)
