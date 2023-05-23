"""Parameter continuation with given function
"""
from typing import Any
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
    y : numpy.ndarray, float, int, list, tuple, or None
        Obtained result of a single step.
    p : dict[str, Any] | None
        Parameter dictionary.
    others : dict[str, Any] | None
        Any other values to return, by default ``None``.
    """

    def __init__(
        self,
        success: bool,
        y: YC | None,
        p: P | None,
        others: Any | None = None,
    ) -> None:
        self.success = success
        self.y = y
        self.p = p
        self.others = others


def continuation(
    fun: Callable[[YC, P | None], ContinuationFunResult],
    y0: YC,
    params: P,
    end_val: float,
    param_key: str,
    resolution: int = 100,
    show_progress: bool = False,
) -> list[tuple[YC, P, dict[str, Any] | None]]:
    """Perform a continuation algorithm to find solutions of a function.

    Parameters:
        fun : callable
            A function that takes arguments `y` and `p` and returns a `ContinuationFunResult`.
        y0 : YC
            Initial value of `y`.
        params : P
            Initial parameter values `p`.
        end_val : float
            The end value of the parameter specified by `param_key`.
        param_key : str
            The key for the parameter that will be varied during the continuation algorithm.
        resolution : int, optional
            The number of steps to perform during the continuation algorithm (default: 100).
        show_progress : bool, optional
            Whether to show progress during the algorithm (default: False).

    Returns:
        list[tuple[YC, P, dict[str, Any] | None]]
            A list of tuples containing the updated `y`, `p`, and additional data (`o`) for each successful step.

    Raises:
        TypeError
            If the return value of `fun` is not an instance of `ContinuationFunResult`,
            or if the types of `y` or `p` are not compatible with the initial values.
    """

    h = (end_val - params[param_key]) / (resolution - 1)

    y = y0.copy() if isinstance(y0, ndarray) else y0
    p = params.copy()

    found: list[tuple[YC, P, dict[str, Any] | None]] = []

    for i in range(resolution):
        ret = fun(y, p)

        if not isinstance(ret, ContinuationFunResult):
            raise TypeError

        if not ret.success:
            break

        y = ret.y
        p = ret.p
        o = ret.others

        if y is None or p is None:
            break

        if not is_type_of(y, type(y0)):
            raise TypeError(type(y), type(y0))
        if not is_type_of(p, type(params)):
            raise TypeError(type(p), type(params))

        if show_progress:
            precision = 10
            show_str: list[str] = ["\tSUCCESS", f"{i+1:0{len(str(resolution))}d}"]
            for val in [y, list(p.values())]:
                if isinstance(val, float):
                    show_str.append(f"{val:+.{precision}f}")
                else:
                    show_str.append(str(val))
            print(" ".join(show_str), end="\r")

        found.append((y, p.copy(), None if o is None else o.copy()))

        if i != resolution - 1:
            p[param_key] += h
    if show_progress:
        print()
    return found
