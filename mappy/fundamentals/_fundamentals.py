"""Fundamental classes and functions
"""
from typing import Generic, Literal, Sequence
from collections.abc import Callable
from functools import wraps
import numpy
from scipy.integrate import solve_ivp

numpy.set_printoptions(precision=12)
import sympy
from ._core import BasicResult
from ..typing import Y, YF, YB, P, Transition


def convert_y_ndarray(y: Y) -> numpy.ndarray:
    """
    Convert the input ``y`` to a NumPy ndarray.

    Parameters
    ----------
    y : int, float, list, tuple, or numpy.ndarray
        Input value to be converted.

    Returns
    -------
    numpy.ndarray
        Converted value as a NumPy ndarray.

    Raises
    ------
    TypeError
        If the type of ``y`` is not supported.

    See Also
    --------
    mappy.revert_y_ndarray : Revert a NumPy ndarray back to its original type.
    mappy.typing.Y : Type for ``y``.

    """
    if isinstance(y, (int, float)):
        return numpy.array([y], dtype=numpy.floating)
    elif isinstance(y, (list, tuple)):
        return numpy.array(y, dtype=numpy.floating)
    elif isinstance(y, numpy.ndarray):
        if y.ndim == 0:
            return numpy.array([y], dtype=numpy.floating)
        else:
            return y.copy()
    else:
        raise TypeError(f"Type of y is not supported: {type(y)}")


def revert_y_ndarray(y: numpy.ndarray, y0: Y) -> Y:
    """
    Revert the NumPy ndarray ``y`` back to its original type.

    Parameters
    ----------
    y : numpy.ndarray
        NumPy ndarray to be reverted.
    y0 : int, float, list, tuple, or numpy.ndarray
        Original value type of ``y``.

    Returns
    -------
    int, float, list, tuple, or numpy.ndarray
        Reverted value of ``y`` with the original type.

    Raises
    ------
    TypeError
        If the type of ``y0`` is not supported.

    See Also
    --------
    mappy.convert_y_ndarray : Convert the input ``y`` to a NumPy ndarray.
    mappy.typing.Y : Type for ``y``.

    """
    if isinstance(y0, (int, float)):
        return y[0].item()
    elif isinstance(y0, (list, tuple)):
        return y.tolist()
    elif isinstance(y0, numpy.ndarray):
        return y.copy()
    else:
        raise TypeError(f"Type of y0 is not supported: {type(y0)}")


class Traj:
    """
    Trajectory of a dynamical system.

    Parameters
    ----------
    sol : numpy.ndarray
        Array representing the trajectory.

    See Also
    --------
    mappy.ModeTraj : Represents a mode trajectory.

    """

    def __init__(self, sol: numpy.ndarray) -> None:
        self.sol = sol


class ModeTraj(Traj):
    """
    Trajectory of a dynamical system with modes.

    Parameters
    ----------
    m0 : str
        Initial mode name.
    m1 : str
        Next mode name.
    mtype : Literal["C", "D"]
        Mode type: "C" for continuous mode, "D" for discrete mode.
    sol : numpy.ndarray
        Array representing the mode trajectory.

    See Also
    --------
    mappy.Traj : Represents a trajectory.
    mappy.Mode : Represents a mode.
    mappy.ContinuousMode : Represents a continuous mode.
    mappy.DiscreteMode : Represents a discrete mode.

    """

    def __init__(
        self, m0: str, m1: str, mtype: Literal["C", "D"], sol: numpy.ndarray
    ) -> None:
        super().__init__(sol)
        self.m0 = m0
        self.m1 = m1
        self.mtype = mtype


class Sol:
    """
    Bundle of initial value, final value, and trajectory.

    Parameters
    ----------
    y0 : Y
        Initial value.
    y1 : Y
        Final value.
    trajs : list[Traj]
        List of trajectories.

    See Also
    --------
    mappy.ModeSol : Represents a mode solution.
    mappy.Traj : Represents a trajectory.
    mappy.typing.Y : Type for ``y0`` and ``y1``.

    """

    def __init__(self, y0: Y, y1: Y, trajs: Sequence[Traj]) -> None:
        self.y0 = y0
        self.y1 = y1
        self.trajs = trajs


class ModeSol(Sol):
    """
    Solution with mode information.

    Parameters
    ----------
    y0 : Y
        Initial value.
    m0 : str
        Initial mode.
    y1 : Y
        Final value.
    m1 : str
        Final mode.
    trajs : list[ModeTraj]
        List of trajectories.

    See Also
    --------
    mappy.Sol : Represents a solution.
    mappy.ModeTraj : Represents a mode trajectory.
    mappy.typing.Y : Type for ``y0`` and ``y1``.

    """

    def __init__(
        self,
        y0: Y,
        m0: str,
        y1: Y,
        m1: str,
        trajs: list[ModeTraj],
    ) -> None:
        self.m0 = m0
        self.m1 = m1
        super().__init__(y0, y1, trajs)


class TransitionKeyError(Exception):
    """Exception raised for an undefined transition.

    Parameters
    ----------
    current_mode : str
        The current mode for which the transition is undefined.

    """

    def __init__(self, current_mode: str) -> None:
        self.current_mode = current_mode

    def __str__(self) -> str:
        return f"[Transition] Transition rule of `{self.current_mode}` is undefined."


class AllModesKeyError(Exception):
    """Exception raised for an undefined mode.

    Parameters
    ----------
    modename : str
        The name of the undefined mode.

    """

    def __init__(self, modename: str) -> None:
        self.modename = modename

    def __str__(self) -> str:
        return f"[All modes] Mode with name `{self.modename}` is undefined."


class NextModeNotFoundError(Exception):
    """Exception raised if a next mode is not found."""

    def __str__(self) -> str:
        return f"[Next mode] Not found next mode. The ODE solver finished with status ``0``."


class ModeStepResult(BasicResult, Generic[Y]):
    """Result of a ``step`` in ``Mode``

    The ModeStepResult class provides the result of ``step`` in the mode.

    Parameters
    ----------
    status : int
        Response status of the continuous mode's solve_ivp method.
        For the discrete mode, use 1.
    y : Y
        Value of the solution after the step.
    tend : float or None, optional
        Value of the time after the step in the continuous-time mode. Default is None.
    i_border : int or None, optional
        Index of the border where the trajectory arrives. Default is None.
    jac : numpy.ndarray, float, or None, optional
        Value of the Jacobian matrix. Default is None.
    hes : numpy.ndarray, float, or None, optional
        Value of the Hessian tensor. Default is None.
    sol : numpy.ndarray or None, optional
        Solution trajectory data of the step. Default is None.

    See Also
    --------
    mappy.Mode : Represents a mode.
    mappy.ContinuousMode : Represents a continuous mode.
    mappy.DiscreteMode : Represents a discrete mode.
    mappy.typing.Y : Type for ``y``.

    """

    def __init__(
        self,
        status: int,
        y: Y,
        tend: float | None = None,
        jac: numpy.ndarray | float | None = None,
        hes: numpy.ndarray | float | None = None,
        i_border: int | None = None,
        sol: numpy.ndarray | None = None,
    ) -> None:
        self.status = status
        self.y = y
        self.i_border = i_border
        self.jac = jac
        self.hes = hes
        self.sol = sol
        self.tend = tend


class Mode(Generic[Y, YF]):
    """Base Class of all modes

    The base class for continuous and discrete modes.

    Parameters
    ----------
    name : str
        Name of the mode.
    fun : Callable
        Evolutional function in the mode.
    mtype : Literal["C", "D"]
        Type of the mode: "C" for continuous, "D" for discrete.
    inTraj : bool, optional
        Flag indicating whether the mode is used in trajectory computation. Default is ``True``.


    See Also
    --------
    mappy.ContinuousMode : Represents a continuous mode.
    mappy.DiscreteMode : Represents a discrete mode.

    """

    def __init__(
        self,
        name: str,
        fun: Callable[[Y, P | None], YF],
        mtype: Literal["C", "D"],
        inTraj: bool = True,
    ) -> None:
        self.name = name
        self.fun = fun
        self.mtype: Literal["C", "D"] = mtype
        self.inTraj = inTraj

        # Jacobian for fun by SymPy
        x_symb = sympy.symbols(" ".join([f"x_{i}" for i in range(self.fun.dom_dim)]))
        p_symb: P | None = None
        p_symb_list: list[sympy.Symbol] | None = None
        if len(self.fun.param_keys) > 0:
            p_symb = {k: sympy.Symbol(k) for k in self.fun.param_keys}
            p_symb_list = list(p_symb.values())
        if hasattr(self.fun, "cod_dim") and self.fun.cod_dim == 1:
            f = sympy.Matrix([fun(x_symb, p_symb)])
        else:
            f = sympy.Matrix(fun(x_symb, p_symb))

        if self.fun.dom_dim == 1:
            jac = sympy.diff(f, x_symb)
        else:
            if not isinstance(f, sympy.Matrix):
                jac = sympy.derive_by_array(f, x_symb)
            else:
                jac = f.jacobian(x_symb)
        if p_symb is None:
            jac_fun = sympy.lambdify([x_symb], jac, "numpy")
        else:
            jac_fun = sympy.lambdify((x_symb, p_symb_list), jac, "numpy")
        if jac_fun is None:
            raise Exception("[JAC_FUN] lambdify returns None")
        self.jac_fun = jac_fun

        # Hessian for fun by SymPy
        if self.fun.dom_dim == 1:
            hess = sympy.diff(jac, x_symb)
        else:
            hess = sympy.derive_by_array(jac, x_symb)
        if p_symb is None:
            hes_fun = sympy.lambdify([x_symb], hess, "numpy")
        else:
            hes_fun = sympy.lambdify((x_symb, p_symb_list), hess, "numpy")
        if hes_fun is None:
            raise Exception("[HES_FUN] lambdify returns None")
        self.hes_fun = hes_fun

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, x):
        return x is self

    def __ne__(self, x):
        return x is not self

    def step(self, y0: Y, params: P | None = None, **options) -> ModeStepResult[YF]:
        """Step to the next mode

        Perform a step in the mode. Default raises ``NotImplemented``.

        Parameters
        ----------
        y0 : Y
            The initial state to pass to the ``fun``.
        params : P or None, optional
            The parameter to pass to ``fun``. Default is None.
        **options
            Additional options for future implementation.

        Returns
        -------
        NotImplemented
            The method needs to be implemented in the derived classes.

        See Also
        --------
        mappy.ModeStepResult : Represents the result of the step.
        mappy.typing.Y : Type for ``y0``.
        mappy.typing.YF : Type for the return value of ``fun``.
        mappy.typing.P : Type for ``params``.

        """

        return NotImplemented


class ContinuousMode(Mode[Y, YF]):
    """Mode for the continuous-time dynamical system

    The class to define the mode for continuous-time dynamical systems.

    Parameters
    ----------
    name : str
        Name of the mode.
    fun : Callable
        Right-hand side of the continuous-time dynamical system. The calling signature is ``fun(y, p | None)``,
        where ``y`` is the state variable and ``p`` is a parameter dictionary.
    borders : list of callables
        List of the border functions to pass to ``solve_ivp`` as events.
        All border functions should be decorated by ``border`` method of the class.
        The calling signature is ``border(y, p | None)``.
    max_interval : float, optional
        Max interval of the time span, by default ``20``.
        The function ``solve_ivp`` of SciPy takes ``t_span = [0, max_interval]``.
    inTraj : bool, optional
        Flag indicating whether the mode is included in the trajectory. Defaults to ``True``.

    """

    def __init__(
        self,
        name: str,
        fun: Callable[[Y, P | None], YF],
        borders: list[Callable[[Y, P | None], float]],
        max_interval: float = 20.0,
        inTraj: bool = True,
    ) -> None:
        super().__init__(name, fun, "C", inTraj)
        self.max_interval = max_interval
        self.borders = borders

        # Jacobian for borders by SymPy
        x_symb = sympy.symbols(" ".join([f"x_{i}" for i in range(self.fun.dom_dim)]))
        p_symb: P | None = None
        p_symb_list: list[sympy.Symbol] | None = None
        if len(self.fun.param_keys) > 0:
            p_symb = {k: sympy.Symbol(k) for k in self.fun.param_keys}
            p_symb_list = list(p_symb.values())

        self.jac_borders = []
        self.hes_borders = []
        for b in self.borders:
            f = b(x_symb, p_symb)

            if self.fun.dom_dim == 1:
                jac = sympy.diff(f, x_symb)
            else:
                jac = sympy.derive_by_array(f, x_symb)
            if p_symb is None:
                jac_fun = sympy.lambdify([x_symb], jac, "numpy")
            else:
                jac_fun = sympy.lambdify((x_symb, p_symb_list), jac, "numpy")
            if jac_fun is None:
                raise Exception("[JAC_BORDERS] lambdify returns None")
            self.jac_borders.append(jac_fun)

            # Hessian
            if self.fun.dom_dim == 1:
                hess = sympy.diff(jac, x_symb)
            else:
                hess = sympy.derive_by_array(jac, x_symb)
            if p_symb is None:
                hes_fun = sympy.lambdify([x_symb], hess, "numpy")
            else:
                hes_fun = sympy.lambdify((x_symb, p_symb_list), hess, "numpy")
            if hes_fun is None:
                raise Exception("[HES_BORDERS] lambdify returns None")
            self.hes_borders.append(hes_fun)

    def __ode_jac(
        self,
        y: numpy.ndarray,
        params: P | None = None,
        params_arr: numpy.ndarray | None = None,
    ) -> numpy.ndarray:
        dim = self.fun.dom_dim
        dydy0 = y[dim : dim + (dim**2)].reshape((dim, dim), order="F")

        ode = lambda y: self.fun(y, params)
        if params_arr is None:
            ode_jac = self.jac_fun
        else:
            ode_jac = lambda y: self.jac_fun(y, params_arr)

        deriv = numpy.empty(dim + (dim**2))
        deriv[0:dim] = ode(y[0:dim])
        jac = ode_jac(y[0:dim])

        deriv[dim : dim + (dim**2)] = (jac @ dydy0).flatten(order="F")
        return deriv

    def __ode_hes(
        self,
        y: numpy.ndarray,
        params: P | None = None,
        params_arr: numpy.ndarray | None = None,
    ) -> numpy.ndarray:
        dim = self.fun.dom_dim

        dydy0 = y[dim : dim + (dim**2)].reshape((dim, dim), order="F")
        ui, uj = numpy.triu_indices(dim)
        d2ydy02 = numpy.empty(dim**3).reshape(dim, dim, dim)
        d2ydy02[ui, uj] = y[
            dim + (dim**2) : dim + (dim**2) + (dim**2 * (dim + 1) // 2)
        ].reshape(dim * (dim + 1) // 2, dim)
        d2ydy02[uj, ui] = d2ydy02[ui, uj].copy()
        d2ydy02 = d2ydy02.transpose(0, 2, 1)

        ode = lambda y: self.fun(y, params)
        if params_arr is None:
            ode_jac = self.jac_fun
            ode_hes = self.hes_fun
        else:
            ode_jac = lambda y: self.jac_fun(y, params_arr)
            ode_hes = lambda y: self.hes_fun(y, params_arr)

        deriv = numpy.empty(dim + (dim**2) + (dim**2 * (dim + 1) // 2))
        # ODE
        deriv[0:dim] = ode(y[0:dim])

        # Jaboain
        jac = ode_jac(y[0:dim])
        deriv[dim : dim + (dim**2)] = (jac @ dydy0).flatten(order="F")

        # Hessian
        hes = ode_hes(y[0:dim])
        deriv[dim + (dim**2) : dim + (dim**2) + (dim**2 * (dim + 1) // 2)] = (
            ((hes @ dydy0).T @ dydy0 + jac @ d2ydy02)
            .transpose(0, 2, 1)[ui, uj]
            .flatten()
        )
        return deriv

    @classmethod
    def function(cls, dimension: int, param_keys: list[str] = []) -> Callable:
        """Decorator for ``fun`` in ``ContinuousMode``

        The method provides the decorator for ``fun`` in ``ContinuousMode``.

        Parameters
        ----------
        dimension : int
            Dimension of the state space of the system.
        param_keys : list[str], optional
            List of keys for the parameters used in `fun`, by default `[]`.

        Returns
        -------
        Callable
            Decorated `fun` function compatible with `fun` in `ContinuousMode`

        See also
        --------
        mappy.ContinuousMode

        Examples
        --------
        The Izhikevich neuron model includes a system of ODEs
        defined in the 2-dimensional state space :math:`\\R^2`.

        .. math::

            \\deriv{v}{t} &= 0.04 v^2 + 5 v + 140 - u + I, \\\\
            \\deriv{u}{t} &= a(bv-u)

        The following code defines the system as a function compatible with ``ContinuousMode``.

        .. code-block:: python

            from mappy import ContinuousMode as CM

            @CM.function(dimension = 2, param_keys = ['a', 'b', 'I'])
            def izhikevich (y, param):
                v, u = y
                a = param['a']
                b = param['b']
                I = param['I']

                return np.array([
                    0.04 * (v ** 2) + 5.0 * v + 140.0 - u + I,
                    a * (b * v - u)
                ])

            # `izhikevich` is compatible with ContinuousMode class
            mode1 = CM(name = 'mode1', fun = izhikevich)

        Without the decorator, the function ``izhikevich`` becomes to be compatible if it has the special attributes
        as follows.

        .. code-block:: python

            def izhikevich (y, param):
                v, u = y
                a = param['a']
                b = param['b']
                I = param['I']

                return np.array([
                    0.04 * (v ** 2) + 5.0 * v + 140.0 - u + I,
                    a * (b * v - u)
                ])
            izhikevich.dom_dim = 2
            izhikevich.param_keys = ['a', 'b', 'I']

            # `izhikevich` is compatible with ContinuousMode class
            mode1 = CM(name = 'mode1', fun = izhikevich)
        """

        def _decorator(fun: Callable[[Y, P], YF]) -> Callable[[Y, P], YF]:
            @wraps(fun)
            def _wrapper(y: Y, p: P) -> YF:
                ret = fun(y, p)
                return ret

            setattr(_wrapper, "dom_dim", dimension)
            setattr(_wrapper, "param_keys", param_keys)
            return _wrapper

        return _decorator

    @classmethod
    def border(cls, direction: int = 1) -> Callable:
        """Decorator for the element of ``borders`` in ``ContinuousMode``

        The method providing the decorator for a element of ``borders`` in ``ContinuousMode``.
        The calling signature is ``border(y, p | None)`` and it should be the same one as ``fun``.

        Parameters
        ----------
        direction : int, optional
            Direction of a zero crossing, by default ``1``.
            The value is directly passed to the ``direction`` attribute of the
            ``event`` function, which is the argument of ``solve_ivp``.

        Returns
        -------
        Callable
            Decorated ``border`` function compatible with ``ContinuousMode``

        Examples
        --------
        Assume :math:`a \\in \\R` be a parameter for the border function and
        :math:`\\bm x=(x, y)` is a state vector.
        One can implement the border defined by :math:`y+a=0` as follows.

        .. code-block:: python

            from mappy import ContinuousMode as CM

            @CM.border(direction = 1)
            def my_border1 (y, p):
                return y[0] + p['a']

        The following example with strict type hints is the same as above one.

        .. code-block:: python

            from mappy import ContinuousMode as CM
            from numpy import ndarray

            @CM.border(direction = 1)
            def my_border1 (y: ndarray, p: dict[str, float] | None = None):
                if p is None:
                    raise TypeError('Parameter should not be None')
                return y[0] + p['a']

        Another example below implements the border defined by :math:`y - 10 = 0`,
        which does not require any parameters.

        .. code-block:: python

            @border(direction=1)
            def my_border_fun (y: float, _):
                return y - 10

        .. note::

            In the last example, ``my_border_fun`` does not use parameter.
            However it should implement the argument place to input parameter
            due to the package limitation.
            In such a case, the throwaway variable ``_`` is useful.

        See also
        --------
        mappy.ContinuousMode
        """

        def _decorator(fun: Callable[[Y, P | None], YB]) -> Callable[[Y, P | None], YB]:
            @wraps(fun)
            def _wrapper(y: Y, p: P | None) -> YB:
                ret = fun(y, p)
                return ret

            setattr(_wrapper, "direction", direction)
            return _wrapper

        return _decorator

    def step(
        self, y0: Y, params: P | None = None, calc_jac=True, calc_hes=True, **options
    ) -> ModeStepResult[YF]:
        """Step to the next mode

        Step to the next mode

        Parameters
        ----------
        y0 : numpy.ndarray or float
            The initial state y0 of the system evolution.
        params : dict[str, Any], optional
            Parameters to pass to ``fun`` and ``borders``, by default ``None``.
        calc_jac : Boolean, optional
            Flag to calculate the Jacobian matrix of the map from initial value to the result y, by default ``True``.
        calc_hes : Boolean, optional
            Flag to calculate the Hessian matrix of the map from initial value to the result y, by default ``True``.
            If ``True``, calc_jac is automatically set to ``True``.
        dense_output : Boolean, optional
            Flag to calculate the dense output, by default ``False``.
        **options
            The options of ``solve_ivp`` in ``scipy.integrate``.

        Returns
        -------
        ModeStepResult
            The result of the step

        See also
        --------
        mappy.ModeStepResult : The result of the step
        mappy.typing.Y : Type of the state vector
        mappy.typing.P : Type of the parameter dictionary

        """

        # Replace the function for ODE and borders with the compatible forms
        ode_fun = lambda t, y: self.fun(y, params)
        calc_jac = calc_hes or calc_jac

        borders = []
        for ev in self.borders:
            evi = lambda t, y, ev=ev: ev(y, params)
            evi.terminal = True
            evi.direction = ev.direction
            borders.append(evi)

        if calc_jac:
            params_arr = (
                None
                if params is None or len(self.fun.param_keys) == 0
                else numpy.array([params[k] for k in self.fun.param_keys])
            )
            jac_fun = lambda t, y: self.jac_fun(y, params_arr)
            if calc_hes:
                ode = lambda t, y: self.__ode_hes(y, params, params_arr)
            else:
                ode = lambda t, y: self.__ode_jac(y, params, params_arr)

            # Border derivatives
            devs = []
            for dev in self.jac_borders:
                if params is None:
                    devi = lambda t, y, dev=dev: dev(y)
                else:
                    devi = lambda t, y, dev=dev: dev(y, params_arr)
                devs.append(devi)
            d2evs = []
            for dev in self.hes_borders:
                if params is None:
                    d2evi = lambda t, y, dev=dev: dev(y)
                else:
                    d2evi = lambda t, y, dev=dev: dev(y, params_arr)
                d2evs.append(d2evi)
        else:
            jac_fun = None
            ode = lambda t, y: self.fun(y, params)
            devs = None
            d2evs = None

        i_border: int | None = None
        dim = self.fun.dom_dim

        # Copy initial value
        _y0 = convert_y_ndarray(y0)

        # Append an identity matrix to the initial state if calculate Jacobian matrix
        if calc_jac:
            _y0 = numpy.append(_y0, numpy.eye(dim).flatten())
        if calc_hes:
            _y0 = numpy.append(_y0, numpy.zeros(dim**2 * (dim + 1) // 2))

        ## Main loop: solve initial value problem
        sol = solve_ivp(ode, [0, self.max_interval], _y0, events=borders, **options)

        ## Set values to the result instance
        _y1: YF = sol.y.T[-1][0:dim]

        if calc_jac:  # If calculate Jacobian matrix
            jact = numpy.array(sol.y.T[-1][dim : dim + (dim**2)]).reshape(
                (dim, dim), order="F"
            )
            jac = jact.copy()
        else:
            jact = None
            jac = None

        if calc_hes:
            ui, uj = numpy.triu_indices(dim)
            hest = numpy.empty(dim**3).reshape(dim, dim, dim)
            hest[ui, uj] = numpy.array(
                sol.y.T[-1][
                    dim + (dim**2) : dim + (dim**2) + (dim**2 * (dim + 1) // 2)
                ]
            ).reshape(dim * (dim + 1) // 2, dim)
            hest[uj, ui] = hest[ui, uj].copy()
            hest = hest.transpose(0, 2, 1)
            hes = hest.copy()
        else:
            hest = None
            hes = None
        # border detect
        if sol.status == 1:
            # For each borders
            for i, ev in enumerate(self.borders):
                if len(sol.t_events[i]) != 0:
                    if (
                        jact is not None
                        and jac_fun is not None
                        and devs is not None
                        and d2evs is not None
                    ):
                        dydt = numpy.array(ode_fun(0, _y1), dtype=numpy.floating)
                        dbdy = numpy.array(devs[i](0, _y1), dtype=numpy.floating)
                        dot: numpy.floating = numpy.dot(dbdy, dydt)
                        out = numpy.outer(dydt, dbdy)
                        B = numpy.eye(dim) - out / dot
                        jac = B @ jact

                        if hest is not None:
                            dfdy = numpy.array(jac_fun(0, _y1))
                            d2bdy2 = numpy.array(d2evs[i](0, _y1))

                            dBdy = -1.0 / dot * (
                                numpy.tensordot(dfdy.T, dbdy, axes=0)
                                + numpy.tensordot(dydt, d2bdy2, axes=0)
                            ) + numpy.tensordot(
                                d2bdy2 @ dydt + dbdy @ dfdy, out, axes=0
                            ) / (
                                dot**2
                            )
                            hes = (
                                numpy.einsum("ijk, il, km -> ljm", dBdy, jac, jact)
                                + B
                                @ (
                                    hest
                                    - numpy.tensordot(dbdy @ jact, dfdy @ jact, axes=0)
                                    / dot
                                )
                            ).T
                    i_border = i

        # Make a response instance
        result = ModeStepResult[YF](
            sol.status, _y1, tend=sol.t[-1], jac=jac, hes=hes, i_border=i_border
        )
        if options.get("dense_output") and self.inTraj:
            step = max(int((sol.t[-1] - sol.t[0]) / 1e-2), 10)
            result.sol = sol.sol(numpy.linspace(sol.t[0], sol.t[-1], step))

        return result


class DiscreteMode(Mode[Y, YF]):
    """Mode of the discrete-time dynamical system

    Mode of the discrete-time dynamical system

    Parameters
    ----------
    name : str
        Name of the mode.
    fun : Callable
        Right-hand side of the discrete-time dynamical system. The calling signature is fun(y).
    inTraj : bool, optional
        If ``True``, the mode is included in the trajectory, by default ``False``.

    See Also
    --------
    mappy.typing.Y : Type of the domain of the function ``fun``.
    mappy.typing.YF : Type of the codomain of the function ``fun``.
    mappy.typing.P : Type of the parameters of the function ``fun``.

    """

    def __init__(
        self, name: str, fun: Callable[[Y, P | None], YF], inTraj: bool = False
    ) -> None:
        super().__init__(name, fun, "D", inTraj)

    @classmethod
    def function(
        cls, domain_dimension: int, codomain_dimension: int, param_keys: list[str] = []
    ) -> Callable:
        """Decorator for ``fun`` in ``DiscreteMode``

        Decorator for ``fun`` in ``DiscreteMode``

        Parameters
        ----------
        domain_dimension : int
            Dimension of the domain of the function ``fun``.
        codomain_dimension : int
            Dimension of the codomain of the function ``fun``.

        Returns
        -------
        Callable
            Decorated function compatible with ``DiscreteMode``

        See Also
        --------
        mappy.DiscreteMode

        Examples
        --------
        The Henon map is a discrete-time dynamical system defined in
        the 2-dimensional Euclidean space.

        .. math::
            x_{n+1} &= 1 - ax_n^2 + y_n, \\\\
            y_{n+1} &= bx_n

        The following code defines the Henon map as a function compatible with ``DiscreteMode``.

        .. code-block:: python

            from mappy import DiscreteMode as DM

            @DM.function(domain_dimension = 2, codomain_dimension = 2, param_keys = ["a", "b"])
            def henon(vect_x, param):
                x, y = vect_x
                a = param["a"]
                b = param["b"]

                return numpy.array([1 - a * x**2 + y, b * x])

            # `henon` is compatible with `DiscreteMode` class
            mode1 = DM(name = "mode1", fun = henon)

        Without the decorator, the function ``henon`` becomes to be compatible if it has the special attributes
        as follows.

        .. code-block:: python

            def henon(vect_x, param):
                x, y = vect_x
                a = param["a"]
                b = param["b"]

                return numpy.array([1 - a * x**2 + y, b * x])

            henon.dom_dim = 2
            henon.cod_dim = 2
            henon.param_keys = ["a", "b"]

            # `henon` is compatible with `DiscreteMode` class
            mode1 = DM(name = "mode1", fun = henon)

        """

        def _decorator(fun: Callable[[Y, P], YF]) -> Callable[[Y, P], YF]:
            @wraps(fun)
            def _wrapper(y: Y, p: P) -> YF:
                ret = fun(y, p)
                return ret

            setattr(_wrapper, "dom_dim", domain_dimension)
            setattr(_wrapper, "cod_dim", codomain_dimension)
            setattr(_wrapper, "param_keys", param_keys)
            return _wrapper

        return _decorator

    def step(
        self, y0: Y, params: P | None = None, calc_jac=True, calc_hes=True, **options
    ) -> ModeStepResult[YF]:
        """Step to the next mode

        Step to the next mode

        Parameters
        ----------
        y0 : numpy.ndarray or float
            The initial state y0 of the system evolution.
        params : Any, optional
            Arguments to pass to ``fun``, by default None.
        calc_jac : Boolean, optional
            Flag to calculate the Jacobian matrix of the map from initial value to the result y, by default ``True``.
        calc_hes : Boolean, optional
            Flag to calculate the Hessian matrix of the map from initial value to the result y, by default ``True``.
            If ``True``, calc_jac is automatically set to ``True``.
        **options
            For future implementation.

        Returns
        -------
        ModeStepResult
            Result of the step.


        See Also
        --------
        mappy.ModeStepResult
        mappy.typing.Y : Type of the domain of the function ``fun``.
        mappy.typing.YF : Type of the codomain of the function ``fun``.
        mappy.typing.P : Type of the parameters of the function ``fun``.

        """
        ## Setup
        _y0 = convert_y_ndarray(y0)
        i_border: int | None = None
        jac = None
        hes = None
        calc_jac = calc_hes or calc_jac

        # Convert functions into the general form
        if calc_jac:
            params_arr = (
                None
                if params is None or len(self.fun.param_keys) == 0
                else numpy.array([params[k] for k in self.fun.param_keys])
            )
            if params_arr is None:
                jac_fun = self.jac_fun
                hes_fun = self.hes_fun
            else:
                jac_fun = lambda y: self.jac_fun(y, params_arr)
                hes_fun = lambda y: self.hes_fun(y, params_arr)
            if calc_hes:
                mapT = lambda n, y: numpy.hstack(
                    (
                        self.fun(y, params),
                        numpy.array(jac_fun(y)).flatten(order="F"),
                        numpy.array(hes_fun(y)).flatten(order="F"),
                    )
                )
            else:
                mapT = lambda n, y: numpy.append(
                    self.fun(y, params), numpy.array(jac_fun(y)).flatten(order="F")
                )
        else:
            mapT = lambda n, y: numpy.array(self.fun(y, params))

        ## Calc y1
        y1 = self.fun(y0, params)

        ## Calc derivatives
        if calc_jac or calc_hes:
            sol = mapT(0, _y0)

            cod_dim: int = self.fun.cod_dim
            dom_dim: int = self.fun.dom_dim

            if calc_jac:
                af = cod_dim
                at = af + (dom_dim * cod_dim)
                jac = sol[af:at].reshape((cod_dim, dom_dim), order="F")
            if calc_hes:
                af = cod_dim + (dom_dim * cod_dim)
                at = af + (dom_dim * cod_dim) * dom_dim
                hes = sol[af:at].reshape((dom_dim, cod_dim, dom_dim), order="F")

        result = ModeStepResult[YF](status=1, y=y1, jac=jac, hes=hes, i_border=i_border)

        if options.get("dense_output") and self.inTraj:
            result.sol = numpy.vstack([_y0, convert_y_ndarray(y1)]).T

        return result


class DiffeomorphismResult(BasicResult, Generic[Y]):
    """Result of ``solve_ivp``

    Parameters
    ----------
    y : Y
        Value of the solution at the final time.
    jac : numpy.ndarray | float | None, optional
        Value of the Jacobian matrix at the final time, by default ``None``.
    eigvals : Y | None, optional
        Eigenvalues of the Jacobian at the final time, by default ``None``.
    eigvecs : numpy.ndarray | float | None, optional
        Eigenvectors of the Jacobian at the final time, by default ``None``.
    hes : numpy.ndarray | float | None, optional
        Value of the Hessian matrix at the final time, by default ``None``.
    sol : Sol | None, optional
        Solution trajectory, by default ``None``.

    See Also
    --------
    mappy.Diffeomorphism : Class of the diffeomorphism.
    mappy.typing.Y : Type of the domain of the function ``fun``.
    mappy.Sol : Class of the solution trajectory.

    """

    def __init__(
        self,
        y: Y,
        jac: numpy.ndarray | float | None = None,
        eigvals: Y | None = None,
        eigvecs: numpy.ndarray | float | None = None,
        hes: numpy.ndarray | float | None = None,
        sol: Sol | None = None,
    ) -> None:
        self.y = y
        self.jac = jac
        self.hes = hes
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.sol = sol


class SolveIvbmpResult(DiffeomorphismResult, Generic[Y]):
    """Result of ``solve_ivbmp``

    Result of ``solve_ivbmp``

    Parameters
    ----------
    y : Y
        Value of the solution at the final time.
    trans_history : list[str]
        List of the transition history.
    jac : numpy.ndarray | float | None, optional
        Value of the Jacobian matrix at the final time, by default ``None``.
    eigvals : Y | None, optional
        Eigenvalues of the Jacobian at the final time, by default ``None``.
    eigvecs : numpy.ndarray | float | None, optional
        Eigenvectors of the Jacobian at the final time, by default ``None``.
    hes : numpy.ndarray | float | None, optional
        Value of the Hessian matrix at the final time, by default ``None``.
    sol : ModeSol | None, optional
        Solution trajectory, by default ``None``.

    See Also
    --------
    mappy.Diffeomorphism : Class of the diffeomorphism.
    mappy.typing.Y : Type of the domain of the function ``fun``.
    mappy.ModeSol : Class of the solution trajectory with mode.
    mappy.typing.Y : Type of the domain of the function ``fun``.

    """

    def __init__(
        self,
        y: Y,
        trans_history: list[str],
        jac: numpy.ndarray | float | None = None,
        eigvals: Y | None = None,
        eigvecs: numpy.ndarray | float | None = None,
        hes: numpy.ndarray | float | None = None,
        sol: ModeSol | None = None,
    ) -> None:
        super().__init__(y, jac, eigvals, eigvecs, hes, sol)
        self.trans_history = trans_history


class SomeJacUndefined(Exception):
    """Exception that some of mode dose not enable Jacobian matrix calculation."""

    def __str__(self) -> str:
        return "Some mode does not implement Jacobian matrix calculation."


class SomeHesUndefined(Exception):
    """Exception that some of mode dose not enable Hessian matrix calculation."""

    def __str__(self) -> str:
        return "Some mode does not implement Hessian tensor calculation."


def solve_ivbmp(
    y0: Y,
    all_modes: tuple[Mode, ...],
    trans: dict[str, str | list[str]],
    m0: str,
    m1: str | list[str] | None = None,
    calc_jac: bool = True,
    calc_hes: bool = False,
    params: P | None = None,
    rtol: float = 1e-6,
    map_count: int = 1,
    dense_output: bool = False,
) -> SolveIvbmpResult[Y]:
    """Solve the initial value and boundary modes problem of the hybrid dynamical system

    Solve the initial value and boundary modes problem of the hybrid dynamical system.

    Parameters
    ----------
    y0 : Y
        Initial state of the system.
    all_modes : tuple[Mode, ...]
        Tuple containing all modes of the hybrid dynamical system.
    trans : dict[str, str | list[str]]
        Dictionary defining the mode transitions in the hybrid system.
    m0 : str
        Initial mode name of the system.
    m1 : str | list[str] | None, optional
        Final mode(s) name of the system, by default None.
    calc_jac : bool, optional
        Flag indicating whether to calculate the Jacobian matrix, by default True.
    calc_hes : bool, optional
        Flag indicating whether to calculate the Hessian matrix, by default False.
    params : P | None, optional
        Dictionary of parameter values for the system, by default None.
    rtol : float, optional
        Relative tolerance for the solver, by default 1e-6.
    map_count : int, optional
        Number of mapping iterations to perform, by default 1.
    dense_output : bool, optional
        Flag indicating whether to use dense output, by default False.

    Returns
    -------
    SolveIvbmpResult[Y]
        Object containing the solution of the initial value and boundary modes problem.

    See Also
    --------
    mappy.Mode : Class of the mode.
    mappy.typing.Y : Type of the domain of the function ``fun``.
    mappy.typing.P : Type of the parameter dictionary.

    """

    _y0 = convert_y_ndarray(y0)

    _y1, jac, _eigs, eigv, hes, trans_history, sol, m1 = _exec_calculation(
        _y0,
        map_count,
        calc_jac,
        calc_hes,
        rtol,
        trans,
        all_modes,
        m0,
        m1,
        params,
        dense_output,
    )

    y1 = revert_y_ndarray(_y1, y0)
    eigs = None if _eigs is None else revert_y_ndarray(_eigs, y0)

    return SolveIvbmpResult(
        y1,
        trans_history,
        jac,
        eigs,
        eigv,
        hes,
        None if len(sol) == 0 else ModeSol(_y0, m0, _y1, m1, sol),
    )


class Diffeomorphism(Generic[Y]):
    """Class of the diffeomorphism

    Class of the diffeomorphism.

    Parameters
    ----------
    name : str
        Name of the mode.
    fun : Callable
        Function of the discrete mode.

    See Also
    --------
    mappy.DiscreteMode : Class of the discrete mode.

    """

    def __init__(self, name: str, fun: Callable):
        self.dm = DiscreteMode(name, fun, inTraj=True)

    def dimension(self):
        """Return the dimension of the domain of the function ``fun``.

        Returns
        -------
        int
            Dimension of the domain of the function ``fun``.

        """
        return self.dm.fun.dom_dim

    def image_detail(
        self,
        y0: Y,
        params: P | None = None,
        iterations: int = 1,
        calc_jac: bool = True,
        calc_hes: bool = True,
    ) -> DiffeomorphismResult[Y]:
        """Image of the diffeomorphism with detail information.

        Parameters
        ----------
        y0 : Y
            Value of the domain of the function ``fun``.
        params : P | None, optional
            Value of the parameter dictionary, by default ``None``.
        iterations : int, optional
            Iteration count of the diffeomorphism, by default 1.
        calc_jac : bool, optional
            Flag indicating whether to calculate the Jacobian matrix, by default ``True``.
        calc_hes : bool, optional
            Flag indicating whether to calculate the Hessian matrix, by default ``True``.

        Returns
        -------
        DiffeomorphismResult[Y]
            Object containing the image of the diffeomorphism with detail information.

        Raises
        ------
        ValueError
            If ``iterations`` is less than 1.

        See Also
        --------
        mappy.DiffeomorphismResult : Class of the result of the diffeomorphism.
        mappy.typing.Y : Type of the domain of the function ``fun``.
        mappy.typing.P : Type of the parameter dictionary.

        """
        if iterations < 1:
            raise ValueError("iterations must be greater than or equal to 1.")

        _y0 = convert_y_ndarray(y0)
        _y1, jac, _eigs, eigv, hes, _, sol, _ = _exec_calculation(
            _y0,
            iterations,
            calc_jac,
            calc_hes,
            0,
            {self.dm.name: self.dm.name},
            (self.dm,),
            self.dm.name,
            None,
            params,
            True,
        )

        y1 = revert_y_ndarray(_y1, y0)
        eigs = None if _eigs is None else revert_y_ndarray(_eigs, y0)
        ret = DiffeomorphismResult[Y](
            y1,
            jac,
            eigs,
            eigv,
            hes,
            None if len(sol) == 0 else Sol(_y0, _y1, sol),
        )

        return ret

    def image(
        self,
        y0: Y,
        params: P | None = None,
        iterations: int = 1,
    ) -> Y:
        """Image of the diffeomorphism.

        Parameters
        ----------
        y0 : Y
            Value of the domain of the function ``fun``.
        params : P | None, optional
            Value of the parameter dictionary, by default ``None``.
        iterations : int, optional
            Iteration count of the diffeomorphism, by default 1.

        Returns
        -------
        Y
            Image of the diffeomorphism.

        Raises
        ------
        ValueError
            If ``iterations`` is less than 1.

        See Also
        --------
        mappy.typing.Y : Type of the domain of the function ``fun``.
        mappy.typing.P : Type of the parameter dictionary.

        """

        ret = self.image_detail(y0, params, iterations, False, False)
        return ret.y

    def traj(self, y0: Y, params: P | None = None, iterations: int = 1) -> Sol | None:
        """Trajectory of the diffeomorphism.

        Parameters
        ----------
        y0 : Y
            Value of the domain of the function ``fun``.
        params : P | None, optional
            Value of the parameter dictionary, by default ``None``.
        iterations : int, optional
            Iteration count of the diffeomorphism, by default 1.

        Returns
        -------
        Sol | None
            Trajectory of the diffeomorphism.

        Raises
        ------
        ValueError
            If ``iterations`` is less than 1.

        See Also
        --------
        mappy.typing.Y : Type of the domain of the function ``fun``.
        mappy.typing.P : Type of the parameter dictionary.

        """

        ret = self.image_detail(y0, params, iterations, False, False)
        return ret.sol


class PoincareMap(Diffeomorphism, Generic[Y]):
    """Class of the Poincare map.

    Parameters
    ----------
    all_modes : tuple[Mode, ...]
        Tuple of all modes.
    trans : Transition
        Dictionary of the transition modes.
    m1 : str | list[str] | None, optional
        Name of the final mode, by default ``None``.

    See Also
    --------
    mappy.Diffeomorphism : Class of the diffeomorphism.
    mappy.Mode : Class of the mode.
    mappy.typing.Transition : Class of the transition modes.

    """

    def __init__(
        self,
        all_modes: tuple[Mode, ...],
        trans: Transition,
        m1: str | list[str] | None = None,
        **options,
    ) -> None:
        self.all_modes = all_modes
        self.trans = trans
        self.options = options
        self.m1 = m1

    def dimension(self, m0: str) -> int:
        """Return the dimension of the domain of the function ``fun``.

        Parameters
        ----------
        m0 : str
            Name of the target mode.

        Returns
        -------
        int
            Dimension of the domain of the function ``fun``.

        Raises
        ------
        ValueError
            If ``m0`` is not in ``all_modes``.

        """
        if m0 not in [_m.name for _m in self.all_modes]:
            raise ValueError(f"{m0} is not in all_modes.")
        return self.all_modes[[_m.name for _m in self.all_modes].index(m0)].fun.dom_dim

    def image_detail(
        self,
        y0: Y,
        m0: str,
        params: P | None = None,
        iterations: int = 1,
        calc_jac: bool = True,
        calc_hes: bool = True,
    ) -> SolveIvbmpResult[Y]:
        """Return the image of the Poincare map with detail information.

        Parameters
        ----------
        y0 : Y
            Value of the domain of the function ``fun``.
        m0 : str
            Name of the initial mode.
        params : P | None, optional
            Value of the parameter dictionary, by default ``None``.
        iterations : int, optional
            Iteration count of the Poincare map, by default 1.
        calc_jac : bool, optional
            Whether to calculate the Jacobian matrix, by default ``True``.
        calc_hes : bool, optional
            Whether to calculate the Hessian matrix, by default ``True``.

        Returns
        -------
        SolveIvbmpResult[Y]
            Result of the Poincare map.
        """
        slv = solve_poincare_map(
            y0,
            self.all_modes,
            self.trans,
            m0,
            self.m1,
            calc_jac=calc_jac,
            calc_hes=calc_hes,
            params=params,
            map_count=iterations,
            **self.options,
        )
        return slv

    def image(
        self,
        y0: Y,
        m0: str,
        params: P | None = None,
        iterations: int = 1,
    ) -> Y:
        """Image of the Poincare map.

        Parameters
        ----------
        y0 : Y
            Value of the domain of the function ``fun``.
        m0 : str
            Name of the initial mode.
        params : P | None, optional
            Value of the parameter dictionary, by default ``None``.
        iterations : int, optional
            Iteration count of the Poincare map, by default 1.

        Returns
        -------
        Y
            Image of the Poincare map.
        """
        slv = self.image_detail(y0, m0, params, iterations, False, False)
        return slv.y

    def traj(
        self,
        y0: Y,
        m0: str,
        params: P | None = None,
        iterations: int = 1,
    ) -> Sol | None:
        """Trajectory of the Poincare map.

        Parameters
        ----------
        y0 : Y
            Value of the domain of the function ``fun``.
        m0 : str
            Name of the mode.
        params : P | None, optional
            Value of the parameter dictionary, by default ``None``.
        iterations : int, optional
            Iteration count of the Poincare map, by default 1.

        Returns
        -------
        Sol | None
            Trajectory of the Poincare map.
        """

        slv = self.image_detail(y0, m0, params, iterations, False, False)
        return slv.sol


def solve_poincare_map(
    y0: Y,
    all_modes: tuple[Mode, ...],
    trans: Transition,
    m0: str,
    m1: str | list[str] | None = None,
    calc_jac: bool = True,
    calc_hes: bool = False,
    params: P | None = None,
    rtol: float = 1e-6,
    map_count: int = 1,
    dense_output: bool = True,
) -> SolveIvbmpResult[Y]:
    """Solve the Poincare map.

    Parameters
    ----------
    y0 : Y
        Value of the domain of the function ``fun``.
    all_modes : tuple[Mode, ...]
        Tuple of all modes.
    trans : Transition
        Dictionary of the transition modes.
    m0 : str
        Name of the initial mode.
    m1 : str | list[str] | None, optional
        Name(s) of the final mode, by default ``None``.
    calc_jac : bool, optional
        Whether to calculate the Jacobian matrix, by default ``True``.
    calc_hes : bool, optional
        Whether to calculate the Hessian matrix, by default ``False``.
    params : P | None, optional
        Value of the parameter dictionary, by default ``None``.
    rtol : float, optional
        Relative tolerance, by default 1e-6.
    map_count : int, optional
        Iteration count of the Poincare map, by default 1.
    dense_output : bool, optional
        Whether to calculate the dense output, by default ``True``.


    Returns
    -------
    SolveIvbmpResult[Y]
        Result of the Poincare map.

    """
    _y0 = convert_y_ndarray(y0)

    _y1, jac, _eigs, eigv, hes, trans_history, sol, m1 = _exec_calculation(
        _y0,
        map_count,
        calc_jac,
        calc_hes,
        rtol,
        trans,
        all_modes,
        m0,
        m1,
        params,
        dense_output,
    )

    y1 = revert_y_ndarray(_y1, y0)

    eigs = None if _eigs is None else revert_y_ndarray(_eigs, y0)

    return SolveIvbmpResult[Y](
        y1,
        trans_history,
        jac,
        eigs,
        eigv,
        hes,
        None if len(sol) == 0 else ModeSol(_y0, m0, _y1, m1, sol),
    )


def _exec_calculation(
    y0: numpy.ndarray,
    map_count: int,
    calc_jac: bool,
    calc_hes: bool,
    rtol: float,
    trans: dict[str, str | list[str]] | dict[str, str] | dict[str, list[str]],
    all_modes: tuple[Mode, ...],
    initial_mode: str,
    m1: str | list[str] | None,
    params: P | None,
    dense_output: bool,
):
    mdi = [m.name for m in all_modes]
    trans_history: list[str] = []
    try:
        _ = mdi.index(initial_mode)
    except KeyError as e:
        raise AllModesKeyError(initial_mode) from e
    current_mode: Mode = all_modes[mdi.index(initial_mode)]
    trans_history.append(initial_mode)

    if m1 is None:
        m1 = [initial_mode]
    elif isinstance(m1, str):
        m1 = [m1]
    else:
        pass

    for em in m1:
        try:
            _ = mdi.index(em)
        except KeyError as e:
            raise AllModesKeyError(em) from e

    jac = None
    hes = None
    count = 0

    _y0 = y0.copy()
    sol = []

    for _ in range(map_count):
        while 1:
            result = current_mode.step(
                _y0,
                params=params,
                calc_jac=calc_jac,
                calc_hes=calc_hes,
                rtol=rtol,
                dense_output=dense_output,
            )
            if result.status == 0:
                raise NextModeNotFoundError

            _y0 = convert_y_ndarray(result.y)

            if calc_jac:
                if result.jac is not None:
                    if jac is None:
                        jac = numpy.eye(current_mode.fun.dom_dim)
                    jacn = numpy.array(result.jac)
                    if calc_hes:
                        if result.hes is not None:
                            if hes is None:
                                hes = numpy.zeros(
                                    ([current_mode.fun.dom_dim] * 3),
                                )

                            hesn = result.hes
                            hes = (hesn @ jac).T @ jac + jacn @ hes
                        else:
                            raise SomeHesUndefined
                    jac = jacn @ jac
                else:
                    raise SomeJacUndefined

            try:
                next = trans[current_mode.name]
                if isinstance(next, list):
                    if result.i_border is None:
                        raise KeyError
                    next = next[result.i_border]
            except KeyError as e:
                raise TransitionKeyError(current_mode.name) from e

            if result.sol is not None and current_mode.inTraj:
                sol.append(
                    ModeTraj(current_mode.name, next, current_mode.mtype, result.sol)
                )

            try:
                current_mode = all_modes[mdi.index(next)]
            except KeyError as e:
                raise AllModesKeyError(next)
            trans_history.append(next)

            count += 1
            if current_mode.name in m1:
                break

    eigs, eigv = _calc_eig_from_jac(jac)
    _hes = _convert_hes_ndarray(hes)

    return (_y0, jac, eigs, eigv, _hes, trans_history, sol, current_mode.name)


def _calc_eig_from_jac(jac: numpy.ndarray | None):
    eigs = None
    eigv = None

    if jac is not None:
        if jac.size == 1:
            jac = convert_y_ndarray(float(jac))
            eigs = jac
        else:
            jac = numpy.squeeze(jac)
            try:  # If Jac is square
                eigs, eigv = numpy.linalg.eig(jac)
            except:
                pass

    return (eigs, eigv)


def _convert_hes_ndarray(hes: numpy.ndarray | None):
    if hes is not None:
        if hes.size == 1:
            hes = convert_y_ndarray(float(hes))
        else:
            hes = numpy.squeeze(hes)
    return hes
