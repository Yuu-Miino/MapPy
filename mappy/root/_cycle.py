"""Cycle (periodic point)
"""
from typing import Generic, Any
from collections.abc import Callable
from mappy import (
    Diffeomorphism,
    PoincareMap,
    BasicResult,
    convert_y_ndarray,
    revert_y_ndarray,
)
import numpy
from scipy.optimize import root, OptimizeResult
from ..typing import Y, P
from ..continuation import ContinuationFunResult, continuation


def _cond_cycle(
    pmap: Callable[[numpy.ndarray, P | None, int], numpy.ndarray],
    y0: numpy.ndarray,
    params: P | None,
    period: int,
) -> numpy.ndarray:
    y1 = pmap(y0, params, period)
    return y1 - y0


class FindCycleResult(BasicResult, Generic[Y]):
    """
    Result of finding a periodic cycle

    Parameters
    ----------
    success : bool
        ``True`` if finding is success, or ``False`` otherwise.
    y : numpy.ndarray, float, or None
        Value of ``y`` if available.
    eigvals : numpy.ndarray, float, or None
        Eigenvalues of a diffeomorphism at ``y``, if available.
    eigvecs : numpy.ndarray or None
        Eigenvectors corresponding to ``eigvals`` if available.
    itr : int
        Count of iterations elapsed in the method.
    err : numpy.ndarray
        :math:`T(y) - y` in vector form, where :math:`T` is a diffeomorphism.
    """

    def __init__(
        self,
        itr: int,
        err: Y,
        success: bool = False,
        y: Y | None = None,
        eigvals: Y | None = None,
        eigvecs: numpy.ndarray | None = None,
    ) -> None:
        self.success = success
        self.y = y
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.itr = itr
        self.err = err


def find_cycle(
    diff: Diffeomorphism,
    y0: Y,
    params: P | None = None,
    period: int = 1,
    m0: str | None = None,
) -> FindCycleResult[Y]:
    """
    Find a periodic cycle of given map

    Parameters
    ----------
    diff : Diffeomorphism
        Diffeomorphism representing the map.
    y0 : Y
        Initial value for a periodic cycle.
    params : P or None, optional
        Parameter array to pass to ``diff``, by default ``None``.
    period : int, optional
        Period of the target periodic cycle, by default ``1``.
    m0 : str or None, optional
        Initial condition identifier for PoincareMap, by default ``None``.

    Returns
    -------
    FindCycleResult[Y]
        Instance containing the result of finding calculation.

    See also
    --------
    mappy.Diffeomorphism : Class representing a diffeomorphism.
    mappy.PoincareMap : Class representing a Poincare map.
    mappy.typing.Y : Type for ``y0`` parameter.
    mappy.typing.P : Type for ``params`` parameter.
    """
    if isinstance(diff, PoincareMap):
        if m0 is None:
            raise ValueError("m0 must be specified for PoincareMap")
        f = lambda y, p, n: diff.image(y, m0, p, n)
        fd = lambda y, p, n: diff.image_detail(y, m0, p, n, True, True)
    else:
        f = lambda y, p, n: diff.image(y, p, n)
        fd = lambda y, p, n: diff.image_detail(y, p, n, True, True)

    objective_fun = lambda y: _cond_cycle(f, y, params, period)

    _y0 = convert_y_ndarray(y0)
    rt: OptimizeResult = root(objective_fun, _y0)

    y1, eigvals, eigvecs = None, None, None
    err = revert_y_ndarray(rt.fun, y0)
    if rt.success:
        _y1 = rt.x

        jac = fd(_y1, params, period).jac

        if jac is not None:
            if isinstance(jac, numpy.ndarray):
                _eigvals, eigvecs = numpy.linalg.eig(jac)
            else:
                _eigvals = numpy.array([jac])
            eigvals = revert_y_ndarray(_eigvals, y0)

        y1 = revert_y_ndarray(_y1, y0)

    return FindCycleResult[Y](
        success=rt.success, y=y1, eigvals=eigvals, eigvecs=eigvecs, itr=rt.nfev, err=err
    )


def trace_cycle(
    diff: Diffeomorphism[Y],
    y0: Y,
    params: P,
    cnt_param_key: str,
    end_val: float,
    resolution: int = 100,
    period: int = 1,
    show_progress: bool = False,
    m0: str | None = None,
) -> list[tuple[numpy.ndarray, P, dict[str, Any] | None]]:
    """
    Trace a periodic cycle using the continuation algorithm.

    Parameters
    ----------
    diff : Diffeomorphism[Y]
        Diffeomorphism representing the map.
    y0 : Y
        Initial value for the periodic cycle.
    params : P
        Initial parameter values.
    cnt_param_key : str
        The key for the parameter that will be varied during the continuation algorithm.
    end_val : float
        The end value of the parameter specified by ``cnt_param_key``.
    resolution : int, optional
        The number of steps to perform during the continuation algorithm (default: 100).
    period : int, optional
        Period of the target periodic cycle (default: 1).
    show_progress : bool, optional
        Whether to show progress during the algorithm (default: ``False``).
    m0 : str or None, optional
        Initial condition identifier for ``PoincareMap``, by default None.

    Returns
    -------
    list[tuple[numpy.ndarray, P, dict[str, Any] | None]]
        A list of tuples containing the updated ``y``, ``p``, and additional data for each successful step.

    See also
    --------
    mappy.Diffeomorphism : Class representing a diffeomorphism.
    mappy.PoincareMap : Class representing a Poincare map.
    mappy.typing.Y : Type for ``y0`` parameter.
    mappy.typing.P : Type for ``params`` parameter.
    """

    def lamb(y: numpy.ndarray, p: P | None):
        ret = find_cycle(diff, y, p, period, m0=m0)
        return ContinuationFunResult(
            ret.success,
            ret.y,
            p,
            {
                "eigvals": ret.eigvals,
                "eigvecs": ret.eigvecs,
                "itr": ret.itr,
                "err": ret.err,
            },
        )

    _y0 = convert_y_ndarray(y0)
    return continuation(
        lamb,
        _y0,
        params,
        end_val,
        param_key=cnt_param_key,
        resolution=resolution,
        show_progress=show_progress,
    )
