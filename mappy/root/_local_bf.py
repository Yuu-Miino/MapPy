"""Local bifurcation set
"""
from typing import TypeVar, Any
from collections.abc import Callable
from mappy import (
    PoincareMap,
    Diffeomorphism,
    DiffeomorphismResult,
    convert_y_ndarray,
    revert_y_ndarray,
)
import numpy
from scipy.optimize import root, OptimizeResult
from ..typing import is_type_of, Y, P
from ..continuation import ContinuationFunResult, continuation
from ._cycle import FindCycleResult

VAR_LBF = TypeVar("VAR_LBF", bound=numpy.ndarray)


def _cond_local_bf(
    pmap: Callable[[numpy.ndarray, P | None], DiffeomorphismResult],
    var: VAR_LBF,
    params: P,
    param_key: str,
    dimension: int,
) -> VAR_LBF:
    y0 = var[0:dimension]
    param, theta = var[dimension:]
    inparams = params.copy()
    inparams[param_key] = param

    res = pmap(y0, inparams)
    det = numpy.linalg.det(res.jac - numpy.exp(1j * theta) * numpy.eye(dimension))

    ret = numpy.empty(
        dimension + 2,
    )
    ret[0:dimension] = res.y - y0
    ret[dimension:] = numpy.real(det), numpy.imag(det)
    return revert_y_ndarray(ret, var)


class FindLocalBfResult(FindCycleResult[Y]):
    def __init__(
        self,
        itr: int,
        err: Y,
        success: bool = False,
        y: Y | None = None,
        eigvals: Y | None = None,
        eigvecs: numpy.ndarray | None = None,
        theta: float | None = None,
        params: P | None = None,
    ) -> None:
        super().__init__(itr, err, success, y, eigvals, eigvecs)
        self.theta = theta
        self.params = params


def find_local_bf(
    diff: Diffeomorphism,
    y0: Y,
    params: P,
    param_key: str,
    theta: float,
    period: int = 1,
    m0: str | None = None,
) -> FindLocalBfResult[Y]:
    f = lambda y, p: diff.image_detail(y0=y, params=p, iterations=period, m0=m0)
    dim = diff.dimension(m0=m0)

    objective_fun = lambda y: _cond_local_bf(
        pmap=f,
        var=y,
        params=params,
        param_key=param_key,
        dimension=dim,
    )

    var = numpy.array(y0).squeeze()
    var = numpy.append(var, [params[param_key], theta])
    rt: OptimizeResult = root(objective_fun, var)

    y1, eigvals, eigvecs = None, None, None
    inparams = params.copy()
    theta1: float | None = None
    err = revert_y_ndarray(rt.fun, y0)
    if rt.success:
        _y1 = rt.x[0:dim]
        param1, theta1 = rt.x[dim:]
        inparams[param_key] = param1

        jac = f(_y1, inparams).jac

        if jac is not None:
            if isinstance(jac, numpy.ndarray):
                _eigvals, eigvecs = numpy.linalg.eig(jac)
            else:
                _eigvals = numpy.array([jac])
            eigvals = revert_y_ndarray(_eigvals, y0)

        y1 = revert_y_ndarray(_y1, y0)

    return FindLocalBfResult[Y](
        success=rt.success,
        y=y1,
        theta=theta1,
        params=inparams,
        eigvals=eigvals,
        eigvecs=eigvecs,
        itr=rt.nfev,
        err=err,
    )


class ParameterKeyError(Exception):
    """Error for conflict of the parameter keys"""

    def __str__(self) -> str:
        return "`param_idx` and `cnt_idx` must be different value."


def trace_local_bf(
    diff: Diffeomorphism[Y],
    y0: Y,
    params: P,
    bf_param_key: str,
    theta: float,
    cnt_param_key: str,
    end_val: float,
    resolution: int = 100,
    period: int = 1,
    show_progress: bool = False,
    m0: str | None = None,
) -> list[tuple[numpy.ndarray, P, dict[str, Any]]]:
    if bf_param_key == cnt_param_key:
        raise ParameterKeyError

    def lamb(y: numpy.ndarray, p: P | None):
        y00, theta0 = y[:-1], y[-1]
        if p is None:
            raise ValueError("Parameter is None but considering bifurcation problem.")
        ret = find_local_bf(diff, y00, p, bf_param_key, theta0, period, m0=m0)
        y1 = None
        if ret.success:
            if ret.y is None or ret.theta is None:
                raise TypeError(ret.y, ret.theta)
            y1 = numpy.append(ret.y, ret.theta)
        p1 = ret.params
        if not is_type_of(p1, type(p)):
            raise TypeError
        return ContinuationFunResult(
            ret.success,
            y1,
            p1,
            {
                "theta": ret.theta,
                "eigvals": ret.eigvals,
                "eigvecs": ret.eigvecs,
                "itr": ret.itr,
                "err": ret.err,
            },
        )

    _y0 = convert_y_ndarray(y0)
    _ret = continuation(
        lamb,
        numpy.append(_y0, theta),
        params,
        end_val,
        param_key=cnt_param_key,
        resolution=resolution,
        show_progress=show_progress,
    )

    ret = [(y, p, info) for y, p, info in _ret if info is not None]

    return ret
