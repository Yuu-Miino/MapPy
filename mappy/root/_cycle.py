"""Cycle (periodic point)
"""
from typing import TypeVar, Generic, TypeAlias, Any
from mappy import PoincareMap
import numpy
from scipy.optimize import root, OptimizeResult
from ..tools import is_type_of, ContinuationFunResult, continuation

Y = TypeVar('Y', numpy.ndarray, float)
P: TypeAlias = dict[str, Any]

def _cond_cycle(
    pmap: PoincareMap,
    y0: Y,
    params: P | None,
    period: int
) -> Y:
    y1 = pmap.image(y0, params, period)
    return y1-y0

class FindCycleResult (Generic[Y]):
    """Result of finding a periodic cycle

    Parameters
    ----------
    success : bool
        True if finding is success, or False otherwise.
    y : numpy.ndarray, float, or None
        Value of `y` if available.
    eigvals : numpy.ndarray, float, or None
        Eigenvalues of the Poincare map at y, if available.
    eigvecs : numpy.ndarray or None
        Eigenvectors corresponding to `eigvals` if available.
    itr : int
        Count of iterations of the method.
    err : numpy.ndarray
        Error of `T(y) - y` in vector form, where `T` is the Poincare map.
    """
    def __init__(
        self,
        itr: int,
        err: numpy.ndarray | float,
        success: bool = False,
        y: Y | None = None,
        eigvals: Y | None = None,
        eigvecs: numpy.ndarray | None = None
    ) -> None:
        self.success = success
        self.y = y
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.itr = itr
        self.err = err
    def __repr__(self) -> str:
        return str({key: val for key, val in self.__dict__.items() if not key.startswith("__")})

def find_cycle(
    poincare_map: PoincareMap,
    y0: Y,
    params: P | None = None,
    period: int = 1
) -> FindCycleResult[Y]:
    """Find a periodic cycle of given map

    Parameters
    ----------
    poincare_map : PoincareMap
        Poincare map.
    y0 : numpy.ndarray or float
        Initial value for a periodic cycle.
    params : numpy.ndarray, float, or None
        Parameter array to pass to `poincare_map`, by default `None`.
    period : int, optional
        Period of the target periodic cycle, by default `1`.

    Returns
    -------
    FindCycleResult
        Instance containing the result of finding calculation

    """

    objective_fun = lambda y: _cond_cycle(poincare_map, y, params, period)

    rt: OptimizeResult = root(objective_fun, y0)

    y1, eigvals, eigvecs = None, None, None
    err = rt.fun
    if rt.success:
        y1 = rt.x

        jac = poincare_map.image_detail(y1, params, period).jac
        if jac is not None:
            if isinstance(jac, numpy.ndarray):
                eigvals, eigvecs = numpy.linalg.eig(jac)
            else:
                eigvals = jac

        if isinstance(y0, float):
            if isinstance(y1, numpy.ndarray) and y1.size == 1:
                y1 = float(y1)
            if isinstance(eigvals, numpy.ndarray) and eigvals.size == 1:
                eigvals = float(eigvals)

        if isinstance(y0, numpy.ndarray):
            if isinstance(y1, float):
                y1 = numpy.array(y1)
            if isinstance(eigvals, float):
                eigvals = numpy.array(eigvals)

        if isinstance(err, numpy.ndarray) and err.size == 1:
            err = float(err)

        if not is_type_of(y1, type(y0)):
            raise TypeError(type(y1), type(y0))

        if not is_type_of(eigvals, type(y0)) and eigvals is not None:
            raise TypeError((type(eigvals), type(y0)))

    return FindCycleResult[Y] (
        success=rt.success,
        y=y1,
        eigvals = eigvals,
        eigvecs = eigvecs,
        itr=rt.nfev,
        err=err
    )

def trace_cycle(
    poincare_map: PoincareMap,
    y0: Y,
    params: P,
    cnt_param_idx: str,
    end_val: float,
    resolution: int = 100,
    period: int = 1,
    show_progress: bool = False
) -> list[dict[str, Y | P ]]:
    def lamb (y: Y, p: P):
        ret = find_cycle(poincare_map, y, p, period)
        return ContinuationFunResult(ret.success, ret.y, p)

    return continuation(
        lamb,
        y0,
        params,
        end_val,
        param_idx=cnt_param_idx,
        resolution=resolution,
        show_progress=show_progress
    )