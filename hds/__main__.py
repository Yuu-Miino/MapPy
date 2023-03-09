import numpy
from .__classes__ import Mode, ContinuousMode, DiscreteMode

class SolveIvbmpResult:
    """Result of `solve_ivbmp`

    Parameters
    ----------
    y : numpy.ndarray or float
        The value of state after mapping.
    jac: numpy.ndarray, float, or None, optional
        Jacobian matrix of the map, by default `None`.
    hes: numpy.ndarray, float, or None, optional
        Hessian tensor of the map, by default `None`.
    """
    def __init__(self,
        y: numpy.ndarray | float,
        jac: numpy.ndarray | float | None = None,
        hes: numpy.ndarray | float | None = None,
    ) -> None:
        self.y = y
        self.jac = jac
        self.hes = hes
    def __repr__(self) -> str:
        return str(self.__dict__)

class SomeJacUndefined(Exception):
    def __str__(self) -> str:
        return "Some mode does not implement Jacobian matrix calculation."

class SomeHesUndefined(Exception):
    def __str__(self) -> str:
        return "Some mode does not implement Hessian tensor calculation."

def solve_ivbmp(
    y0: numpy.ndarray | float,
    initial_mode: Mode,
    end_mode: Mode | None = None,
    calc_jac: bool = True,
    calc_hes: bool = False,
    args = None,
    rtol=1e-6,
    map_count=1
) -> SolveIvbmpResult:
    """Solve the initial value and boundary **modes** problem of the hybrid dynamical system

    Parameters
    ----------
    y0 : numpy.ndarray or float
        The initial value y0.
    initial_mode : Mode
        The initial mode. It also plays the role of Poincaré section since the function solves the boundary modes problem.
    end_mode: Mode or None, optional
        The end mode, by default `None`. If `None`, the end mode in the method is the same as `initial_mode`.
    calc_jac : bool, optional
        Flag to calculate the Jacobian matrix, by default `True`.
    calc_hes : bool, optional
        Flag to calculate the Hessian tensor, by default `True`.
    args : Any, optional
        The parameter to pass to `fun` in all `mode`, by default None.
    rtol : float, optional
        Relative torelance to pass to `solve_ivp`, by default `1e-6`.
    map_count : int, optional
        Count of maps, by default `1`.

    Returns
    -------
    SolveIvbmpResult

    Raises
    ------
    SomeJacUndefined
        Error of not implemented Jacobian matrix calculation.
    SomeHesUndefined
        Error of not implemented Hessian tensor calculation.
    """
    result = None
    current_mode = initial_mode
    if end_mode is None:
        end_mode = initial_mode
    jac = None
    hes = None
    count = 0

    for _ in range(map_count):
        while 1:
            result = current_mode.step(y0, args=[args], rtol=rtol)
            y0 = result.y

            if calc_jac:
                if result.jac is not None:
                    if jac is None:
                        jac = numpy.eye(current_mode.fun.dom_dim)
                    jacn = numpy.array(result.jac)
                    if calc_hes:
                        if result.hes is not None:
                            if hes is None:
                                hes = numpy.zeros((current_mode.fun.dom_dim, current_mode.fun.dom_dim, current_mode.fun.dom_dim))

                            hesn = result.hes
                            hes = (hesn@jac).transpose(2, 1, 0) @ jac + jacn @ hes
                        else:
                            raise SomeHesUndefined
                    jac = jacn @ jac
                else:
                    raise SomeJacUndefined

            if isinstance(current_mode, ContinuousMode):
                if result.i_border is not None:
                    current_mode = current_mode.next[result.i_border]
            elif isinstance(current_mode, DiscreteMode):
                current_mode = current_mode.next
            else:
                pass
            count += 1
            if current_mode == end_mode:
                break

    return SolveIvbmpResult(
        y0,
        float(jac) if jac is not None and jac.size == 1 else (numpy.squeeze(jac) if jac is not None else None),
        float(hes) if hes is not None and hes.size == 1 else (numpy.squeeze(hes) if hes is not None else None)
    )