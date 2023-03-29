"""Local bifurcation set
"""
from typing import TypeVar, TypeAlias, Any
from mappy import PoincareMap
import numpy
from scipy.optimize import root, OptimizeResult
from ..tools import is_type_of, ContinuationFunResult, continuation
from ._cycle import FindCycleResult

Y = TypeVar('Y', numpy.ndarray, float)
P: TypeAlias = dict[str, Any]
V_LBF = TypeVar('V_LBF', bound=numpy.ndarray)

def _cond_local_bf (
    pmap: PoincareMap,
    var: V_LBF,
    params: P,
    param_idx: str,
    period: int = 1
) -> V_LBF:
    y0 = var[0:pmap.dimension]
    param, theta = var[pmap.dimension:]
    inparams = params.copy()
    inparams[param_idx] = param

    res = pmap.image_detail(y0, inparams, iterations=period)
    det = numpy.linalg.det(res.jac - numpy.exp(1j*theta) * numpy.eye(pmap.dimension))

    ret = numpy.empty(pmap.dimension + 2, )
    ret[0:pmap.dimension]  = res.y - y0
    ret[pmap.dimension:] = numpy.real(det), numpy.imag(det)
    if not is_type_of(ret, type(var)):
        raise TypeError(type(ret), type(var))
    return ret

class FindLocalBfResult(FindCycleResult[Y]):
    def __init__(self,
        itr: int,
        err: numpy.ndarray | float,
        success: bool = False,
        y: Y | None = None,
        eigvals: Y | None = None,
        eigvecs: numpy.ndarray | None = None,
        theta: float | None = None,
        params: P | None = None
    ) -> None:
        super().__init__(itr, err, success, y, eigvals, eigvecs)
        self.theta = theta
        self.params = params

def find_local_bf (
    poincare_map: PoincareMap[Y],
    y0: Y,
    params: P,
    param_idx: str,
    theta: float,
    period: int = 1,
) -> FindLocalBfResult[Y]:
    objective_fun = lambda y: _cond_local_bf(
        pmap=poincare_map,
        var=y,
        params=params,
        param_idx=param_idx,
        period=period
    )

    var = numpy.array(y0).squeeze()
    var = numpy.append(var, [params[param_idx], theta])
    rt: OptimizeResult = root(objective_fun, var)

    y1, eigvals, eigvecs = None, None, None
    inparams = params.copy()
    theta1: float | None = None
    err = rt.fun
    if rt.success:
        y1 = rt.x[0:poincare_map.dimension]
        param1, theta1 = rt.x[poincare_map.dimension:]
        inparams[param_idx] = param1

        jac = poincare_map.image_detail(y1, inparams, period).jac
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

    return FindLocalBfResult[Y] (
        success=rt.success,
        y=y1,
        theta=theta1,
        params=inparams,
        eigvals = eigvals,
        eigvecs = eigvecs,
        itr=rt.nfev,
        err=err
    )

class ParameterKeyError(Exception):
    """Error for conflict of the parameter keys
    """
    def __str__(self) -> str:
        return '`param_idx` and `cnt_idx` must be different value.'

def trace_local_bf(
    poincare_map: PoincareMap[Y],
    y0: Y,
    params: P,
    bf_param_idx: str,
    theta: float,
    cnt_param_idx: str,
    end_val: float,
    resolution: int = 100,
    period: int = 1,
    show_progress: bool = False
) -> list[dict[str, numpy.ndarray | P]]:
    if bf_param_idx == cnt_param_idx:
        raise ParameterKeyError

    def lamb (y, p: P):
        if isinstance(y0, float):
            y00 = float(y[0])
        else:
            y00 = numpy.array(y[0:poincare_map.dimension])
        theta0 = float(y[poincare_map.dimension])
        if not is_type_of(y00, type(y0)):
            raise TypeError(type(y00), type(y0))
        ret = find_local_bf(poincare_map, y00, p, bf_param_idx, theta0, period)
        y1 = None
        if ret.success:
            if ret.y is None or ret.theta is None:
                raise TypeError(ret.y, ret.theta)
            y1 = numpy.array(ret.y)
            y1 = numpy.append(y1, ret.theta)
        p1 = ret.params
        if not is_type_of(p1, type(p)):
            raise TypeError
        return ContinuationFunResult(ret.success, y1, p1)

    return continuation(
        lamb,
        numpy.array([y0, theta]),
        params,
        end_val,
        param_idx=cnt_param_idx,
        resolution=resolution,
        show_progress=show_progress
    )