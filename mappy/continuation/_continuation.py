"""Parameter continuation with given function
"""
from collections.abc import Callable
from numpy import ndarray
from ..typing._type import is_type_of, P, YC
from ..fundamentals._core import BasicResult

class ContinuationFunResult(BasicResult):
    """Result of each step in parameter continuation

    The class containing the result of a function, which
    is a sinle step of a parameter continuation.

    Parameters
    ----------
    success : bool
        Whether or not that a single step exited successfully.
    y : numpy.ndarray, float, tuple, or None
        Obtained result of a single step. The value must have the same type as
        the parameter of the step.
    p : dict[str, Any] | None
        Parameter dictionary.
    """
    def __init__(self,
        success: bool, y: YC | None, p: P | None,
    ) -> None:
        self.success = success
        self.y = y
        self.p = p

def continuation(
    fun: Callable[[YC, P | None], ContinuationFunResult],
    y0: YC,
    params: P,
    end_val: float,
    param_idx: str,
    resolution: int = 100,
    show_progress: bool = False,
) -> list[tuple[YC, P]]:
    """Parameter continuation

    The method to operate the parameter continuation of a given `fun`.
    It determines the step size of the continuation from given parameters
    `end_val` and `resolution`.
    In other words, `resolution` is the number of iterations of the continuation.
    Significant alternatives of a `continuation` method in some analytic purposes
    are available in each subpackage, like `mappy.root.trace_cycle`.

    Parameters
    ----------
    fun : Callable
        Function to call in each step.
    y0 : numpy.ndarray, float, or tuple
        Input of `fun` in the initial step.
    params : dict[str, Any]
        Parameter dictionary to pass to `fun`.
    end_val : float
        End value of the continuation.
    param_idx : int, optional
        Index of the parameter to continue in `params`.
    resolution : int, optional
        Resolution of the continuation from the current value to `end_val`, by default `100`.

    See also
    --------
    mappy.root.trace_cycle, mappy.root.trace_local_bf

    Returns
    -------
    list[tuple[y, params]]
        Result list of the tuples containing `y` and `params`.
    """

    h = (end_val-params[param_idx])/(resolution-1)

    y = y0.copy() if isinstance(y0, ndarray) else y0
    p = params.copy()

    found: list[tuple[YC, P]] = []

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

        if not is_type_of(y, type(y0)):
            raise TypeError(type(y), type(y0))
        if not is_type_of(p, type(params)):
            raise TypeError(type(p), type(params))

        if show_progress:
            precision = 10
            show_str: list[str] = ["\tSUCCESS" if ret.success else "FAILURE", f"{i+1:0{len(str(resolution))}d}"]
            for val in [y, list(p.values())]:
                if isinstance(val, float) or len(val) == 1:
                    show_str.append(f"{val:+.{precision}f}")
                else:
                    show_str.append(" ".join(["{:+."+str(precision)+"f}"] * len(val)).format(*val))
            print(" ".join(show_str), end="\r")

        found.append((y, p))

        if i != resolution-1:
            p[param_idx] += h
    if show_progress: print()
    return found
