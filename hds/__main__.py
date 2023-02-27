import numpy
from .__classes__ import Mode, ContinuousMode, DiscreteMode

class FirstReturnMapResult:
    """Result of `first_return_map`

    Parameters
    ----------
    y : numpy.ndarray or float
        The value of state after mapping.
    jac: numpy.ndarray, float, or None, optional
        Jacobian matrix of the map, by default `None`.
    """
    def __init__(self,
        y: numpy.ndarray | float,
        jac: numpy.ndarray | float | None = None
    ) -> None:
        self.y = y
        self.jac = jac
    def __repr__(self) -> str:
        return str(self.__dict__)

class SomeJacUndefined(Exception):
    def __str__(self) -> str:
        return "Some mode does not implement Jacobian matrix calculation."

def first_return_map(
    y0: numpy.ndarray | float,
    initial_mode: Mode,
    calc_jac: bool = True,
    args = None,
    rtol=1e-6,
    map_count=1
) -> FirstReturnMapResult:
    """First return map

    Parameters
    ----------
    y0 : numpy.ndarray or float
        The initial state y0.
    initial_mode : Mode
        The initial mode. It also plays the role of Poincaré section.
    calc_jac : bool, optional
        Flag to calculate the Jacobian matrix, by default `True`.
    args : Any, optional
        The parameter to pass to `fun` in all `mode`, by default None.
    rtol : float, optional
        Relative torelance to pass to `solve_ivp`, by default `1e-6`.
    map_count : int, optional
        Count of maps, by default `1`.

    Returns
    -------
    FirstReturnMapResult

    Raises
    ------
    SomeJacUndefined
        Error of not implemented Jacobian matrix calculation.
    """
    result = None
    current_mode = initial_mode
    finish_mode = initial_mode
    jac = None
    count = 0

    for _ in range(map_count):
        while 1:
            result = current_mode.step(y0, args=[args], rtol=rtol)
            y0 = result.y

            if calc_jac:
                if result.jac is not None:
                    if jac is None:
                        jac = numpy.eye(current_mode.fun.dom_dim)
                    jacn = result.jac
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
            if current_mode == finish_mode:
                break

    return FirstReturnMapResult(y0, float(jac) if jac is not None and jac.size == 1 else jac)