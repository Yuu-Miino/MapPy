"""Cycle (periodic point)
"""
from typing import TypeVar, Generic
from mappy import PoincareMap
import numpy
from scipy.optimize import root, OptimizeResult
from ..tools import is_type_of

Y = TypeVar('Y', numpy.ndarray, float)
P = TypeVar('P', numpy.ndarray, float)

__all__ = [
    "find_cycle",
    "FindCycleResult",
    "ResultDumper"
]

def cond_cycle(
    pmap: PoincareMap,
    y0: Y,
    params: P,
    period: int
) -> Y:
    y1 = pmap.image(y0, params, period)
    return y1-y0

class ResultDumper:
    def __init__(self, dump_keys: list[str]) -> None:
        self.dump_keys = dump_keys

    def dump_values (
        self,
        dump_keys: list[str] = [],
        precision: int =10
    ) -> str:
        """Dump the result values

        Parameters
        ----------
        dump_keys : list of str, optional
            Keys of the results to dump, by default []
        precision : int, optional
            Precision of the numerics of the dumped result, by default 10

        Returns
        -------
        str
            Dumped string
        """
        if len(dump_keys) == 0:
            keys = self.dump_keys.copy()
        else:
            keys = [key for key in dump_keys if key in self.dump_keys]
        out = []
        for k in keys:
            vals = getattr(self, k)
            if vals is None:
                out.append("None")
            elif isinstance(vals, float) or vals.size == 1:
                out.append(f"{vals:+.{precision}f}")
            else:
                out.append(" ".join(["{:+."+str(precision)+"f}"] * vals.size).format(*vals))
        return " ".join(out)

class FindCycleResult (ResultDumper, Generic[Y]):
    def __init__(
        self,
        success: bool,
        y: Y | None,
        eigvals: Y | None,
        eigvecs: numpy.ndarray | None,
        itr: int,
        err: Y,
        dump_keys = ['y', 'eigvals']
    ) -> None:
        self.success = success
        self.y = y
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.itr = itr
        self.err = err
        super().__init__(dump_keys=dump_keys)
    def __repr__(self) -> str:
        return str(self.__dict__)

def find_cycle(
    poincare_map: PoincareMap,
    y0: Y,
    params: P,
    period: int = 1
) -> FindCycleResult[Y]:

    objective_fun = lambda y: cond_cycle(poincare_map, y, params, period)

    rt: OptimizeResult = root(objective_fun, y0)

    eigvals, eigvecs = None, None
    if rt.success:
        jac = poincare_map.image_detail(rt.x, params, period).jac
        if jac is not None:
            if isinstance(jac, numpy.ndarray):
                eigvals, eigvecs = numpy.linalg.eig(jac)
            else:
                eigvals = jac

    x = rt.x

    if isinstance(y0, float) and (isinstance(x, numpy.ndarray) and x.size == 1):
        x = float(x)

    if not is_type_of(x, type(y0)):
        raise TypeError(type(x), type(y0))

    if not is_type_of(eigvals, type(y0)) and eigvals is not None:
        raise TypeError((type(eigvals), type(y0)))

    result = FindCycleResult[Y] (
        success=rt.success,
        y=x,
        eigvals = eigvals,
        eigvecs = eigvecs,
        itr=rt.nfev,
        err=numpy.squeeze(rt.fun)
    )

    return result