"""Fundamental classes and functions
"""
from typing import Generic
from collections.abc import Callable
from functools import wraps
import numpy
from scipy.integrate import solve_ivp, OdeSolution

numpy.set_printoptions(precision=12)
import sympy
from ._core import BasicResult
from ..typing import is_type_of, Y, YF, YB, P


class TransitionKeyError(Exception):
    """Exception for the undefined transition"""

    def __init__(self, current_mode: str) -> None:
        self.current_mode = current_mode

    def __str__(self) -> str:
        return f"[Transition] Transition rule of `{self.current_mode}` is undefined."


class AllModesKeyError(Exception):
    """Exception for the undefined mode"""

    def __init__(self, modename: str) -> None:
        self.modename = modename

    def __str__(self) -> str:
        return f"[All modes] Mode with name `{self.modename}` is undefined."


class NextModeNotFoundError(Exception):
    """Exception that the next mode is not found"""

    def __str__(self) -> str:
        return f"[Next mode] Not found next mode. The ODE solver finished with status ``0``."


class ModeStepResult(BasicResult, Generic[Y]):
    """Result of ``step`` in ``Mode``

    The ModeStepResult class provides the result of ``step`` in the mode.

    Parameters
    ----------
    status : int
        Response status of ``solve_ivp`` for the continuous mode.
        ``1`` for the discrete mode.
    y : numpy.ndarray or float
        Value of the solution after step.
    tend : float or None, optional
        Value of the time after step of the continuous-time mode, by default ``None``.
    i_border : int or None, optional
        Index of the border where the trajectory arrives, by default ``None``.
    jac : numpy.ndarray, float, or None, optional
        Value of the Jacobian matrix, by default ``None``.
    hes : numpy.ndarray, float, or None, optional
        Value of the Hessian tensor, by default ``None``.
    sol : OdeSolution or None, optional
        OdeSolution instance of ``solve_ivp`` in the continuous-time mode, by default ``None``.

    """

    def __init__(
        self,
        status: int,
        y: Y,
        tend: float | None = None,
        jac: numpy.ndarray | float | None = None,
        hes: numpy.ndarray | float | None = None,
        i_border: int | None = None,
        sol: OdeSolution | None = None,
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

    The originator class of continuous and discrete modes.

    Parameters
    ----------
    name : str
        Name of the mode.
    fun : Callable
        Evolutional function in the mode.

    """

    def __init__(self, name: str, fun: Callable[[Y, P | None], YF]) -> None:
        self.name = name
        self.fun = fun

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

        The method to operate the step of the ``mode``.
        The definition depends on the practical type of ``mode``,
        continuous or discrete.

        Parameters
        ----------
        y0 : numpy.ndarray
            The initial state to pass to ``fun``.
        params : Any or None, optional
            The parameter to pass to ``fun``, by default None.
        **options
            For future implementation.

        Returns
        -------
        NotImplementented
        """

        return NotImplemented


class ContinuousMode(Mode[Y, YF]):
    """Mode for the continuos-time dynamical system

    The class to define the mode for continuos-time dynamical systems.

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

    """

    def __init__(
        self,
        name: str,
        fun: Callable[[Y, P | None], YF],
        borders: list[Callable[[Y, P | None], float]],
        max_interval: float = 20.0,
    ) -> None:
        super().__init__(name, fun)
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
        """Decorator for ``fun`` in ``ContinuousTimeMode``

        The method provides the decorator for ``fun`` in ``ContinuousTimeMode``.

        Parameters
        ----------
        dimension : int
            Dimension of the state space of the system.
        param_keys : list[str]
            List of keys for the parameters used in ``fun``, by default ``[]``.

        Returns
        -------
        Callable
            Decorated ``fun`` function compatible with ``fun`` in ``ContinousTimeMode``

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

        The equivalent description for the system by ``mappy`` is as follows.

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
        """Decorator for the element of ``borders`` in ``ContinuousTimeMode``

        The method providing the decorator for a element of ``borders`` in ``ContinuousTimeMode``.
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
            Decorated ``border`` function compatible with ``ContinuousTimeMode``

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

        .. warning::

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
        **options
            The options of ``solve_ivp`` in ``scipy.integrate``.

        Returns
        -------
        ModeStepResult
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
        if isinstance(y0, numpy.ndarray):
            y0in = y0.copy()
        else:
            y0in = y0

        # Append an identity matrix to the initial state if calculate Jacobian matrix
        if calc_jac:
            y0in = numpy.append(y0in, numpy.eye(dim).flatten())
        if calc_hes:
            y0in = numpy.append(y0in, numpy.zeros(dim**2 * (dim + 1) // 2))

        ## Main loop: solve initial value problem
        sol = solve_ivp(ode, [0, self.max_interval], y0in, events=borders, **options)

        ## Set values to the result instance
        y1: YF = sol.y.T[-1][0:dim] if dim != 1 else sol.y.T[-1][0]

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
                        dydt = numpy.array(ode_fun(0, y1), dtype=numpy.float64)
                        dbdy = numpy.array(devs[i](0, y1), dtype=numpy.float64)
                        dot: numpy.float64 = numpy.dot(dbdy, dydt)
                        out = numpy.outer(dydt, dbdy)
                        B = numpy.eye(dim) - out / dot
                        jac = B @ jact

                        if hest is not None:
                            dfdy = numpy.array(jac_fun(0, y1))
                            d2bdy2 = numpy.array(d2evs[i](0, y1))

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
            sol.status, y1, tend=sol.t[-1], jac=jac, hes=hes, i_border=i_border
        )
        if options.get("dense_output"):
            result.sol = sol.sol(numpy.linspace(sol.t[0], sol.t[-1], 100))

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
    """

    def __init__(self, name: str, fun: Callable[[Y, P | None], YF]) -> None:
        super().__init__(name, fun)

    @classmethod
    def function(
        cls, domain_dimension: int, codomain_dimension: int, param_keys: list[str] = []
    ) -> Callable:
        """Decorator for ``fun`` in ``DiscreteTimeMode``

        Decorator for ``fun`` in ``DiscreteTimeMode``

        Parameters
        ----------
        domain_dimension : int
            Dimension of the domain of the function ``fun``.
        codomain_dimension : int
            Dimension of the codomain of the function ``fun``.

        Returns
        -------
        Callable
            Decorated function compatible with ``DiscreteTimeMode``
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

        """
        ## Setup
        y0in = y0 if isinstance(y0, float) else y0.copy()
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

        ## Main part
        sol = mapT(0, y0in)

        cod_dim: int = self.fun.cod_dim
        dom_dim: int = self.fun.dom_dim
        y1 = self.fun(y0in, params)
        if not isinstance(y1, numpy.ndarray):
            y1 = float(y1)

        if isinstance(sol, float) or sol.size == 1:
            tmp = float(sol)
        elif cod_dim == 1:
            tmp = sol[0]
        else:
            tmp = sol[0:cod_dim]

        if not is_type_of(tmp, type(y1)):
            raise TypeError((type(tmp), type(y1)))
        y1 = tmp

        if calc_jac:
            af = cod_dim
            at = af + (dom_dim * cod_dim)
            jac = sol[af:at].reshape((cod_dim, dom_dim), order="F")
        if calc_hes:
            af = cod_dim + (dom_dim * cod_dim)
            at = af + (dom_dim * cod_dim) * dom_dim
            hes = sol[af:at].reshape((dom_dim, cod_dim, dom_dim), order="F")

        result = ModeStepResult[YF](status=1, y=y1, jac=jac, hes=hes, i_border=i_border)

        return result


class SolveIvbmpResult(BasicResult, Generic[Y]):
    """Result of ``solve_ivbmp``

    Result of ``solve_ivbmp``

    Parameters
    ----------
    y : numpy.ndarray or float
        The value of state after mapping.
    trans_history : list[str]
        Transition history of the modes by index of ``all_modes``.
    jac : numpy.ndarray, float, or None, optional
        Jacobian matrix of the map, by default ``None``.
    eigvals : numpy.ndarray, float or None, optional
        Eigenvalues of the Jacobian matrix, by default ``None``.
    eigvecs : numpy.ndarray or None, optional
        Eigenvectors corresponding to ``eigvals``, by default ``None``.
    hes : numpy.ndarray, float, or None, optional
        Hessian tensor of the map, by default ``None``.
    """

    def __init__(
        self,
        y: Y,
        trans_history: list[str],
        jac: numpy.ndarray | float | None = None,
        eigvals: Y | None = None,
        eigvecs: numpy.ndarray | float | None = None,
        hes: numpy.ndarray | float | None = None,
    ) -> None:
        self.y = y
        self.trans_history = trans_history
        self.jac = jac
        self.hes = hes
        self.eigvals = eigvals
        self.eigvecs = eigvecs


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
    initial_mode: str,
    end_mode: str | None = None,
    calc_jac: bool = True,
    calc_hes: bool = False,
    params: P | None = None,
    rtol=1e-6,
    map_count=1,
) -> SolveIvbmpResult[numpy.ndarray] | SolveIvbmpResult[float]:
    """Solve the initial value and boundary modes problem of the hybrid dynamical system

    Solve the initial value and boundary modes problem of the hybrid dynamical system

    Parameters
    ----------
    y0 : numpy.ndarray or float
        Initial value y0.
    all_modes : tuple of Modes
        Set of all modes.
    trans : dict
        Transition function that maps from ``current mode`` to ``next mode``.
    initial_mode : str
        Name of the initial mode.
    end_mode : Mode or None, optional
        Name of the end mode, by default ``None``. If ``None``, the end mode in the method is the same as ``initial_mode``.
    calc_jac : bool, optional
        Flag to calculate the Jacobian matrix, by default ``True``.
    calc_hes : bool, optional
        Flag to calculate the Hessian tensor, by default ``True``.
    params : Parameter, optional
        Parameter to pass to ``fun`` in all ``mode``, by default ``None``.
    rtol : float, optional
        Relative torelance to pass to ``solve_ivp``, by default ``1e-6``.
    map_count : int, optional
        Count of maps, by default ``1``.

    Returns
    -------
    SolveIvbmpResult

    """

    y0in, jac, eigs, eigv, hes, trans_history = _exec_calculation(
        y0,
        map_count,
        calc_jac,
        calc_hes,
        rtol,
        trans,
        all_modes,
        initial_mode,
        end_mode,
        params,
    )

    if isinstance(y0in, numpy.ndarray):
        if isinstance(eigs, float):
            raise TypeError
        return SolveIvbmpResult[numpy.ndarray](
            y0in, trans_history, jac, eigs, eigv, hes
        )
    else:
        if isinstance(eigs, numpy.ndarray):
            raise TypeError
        return SolveIvbmpResult[float](y0in, trans_history, jac, eigs, eigv, hes)


class PoincareMap(Generic[Y]):
    """Construct Poincare map

    Construct Poincare map

    Parameters
    ----------
    all_modes : tuple[Mode, ...]
        Tuple containing all modes.
    trans : dict[str, str  |  list[str]]
        Dictionary defining transition rule.
    initial_mode : str
        Initial mode of the Poincare map.
    calc_jac : bool, optional
        Flag to calculate Jacobian matrix of the map, by default ``False``.
    calc_hes : bool, optional
        Flag to calculate Hessian matrix of the map, by default ``False``.
    """

    def __init__(
        self,
        all_modes: tuple[Mode, ...],
        trans: dict[str, str | list[str]],
        initial_mode: str,
        calc_jac: bool = False,
        calc_hes: bool = False,
        **options,
    ) -> None:
        self.all_modes = all_modes
        self.trans = trans
        self.initial_mode = initial_mode
        self.calc_jac = calc_jac
        self.calc_hes = calc_hes
        self.options = options
        all_modes_name = [m.name for m in all_modes]
        if initial_mode not in all_modes_name:
            raise AllModesKeyError(initial_mode)
        else:
            self.dimension = all_modes[all_modes_name.index(initial_mode)].fun.dom_dim

    def image_detail(
        self, y0: Y, params: P | None, iterations: int = 1
    ) -> SolveIvbmpResult[Y]:
        """Calculate image of the Poincare map with detailed information

        Parameters
        ----------
        y0 : numpy.ndarray or float
            Initial state.
        iterations : int, optional
            Count of iterations of the map, by default ``1``.

        Returns
        -------
        SolveIvbmpResult
            Result of calculation.
        """
        slv = solve_poincare_map(
            y0,
            self.all_modes,
            self.trans,
            self.initial_mode,
            calc_jac=self.calc_jac,
            calc_hes=self.calc_hes,
            params=params,
            map_count=iterations,
            **self.options,
        )
        return slv

    def image(self, y0: Y, params: P | None = None, iterations: int = 1) -> Y:
        """Calculate image of the Poincare map

        Parameters
        ----------
        y0 : numpy.ndarray | float
            Element to calculate the image under the Poincare map.
        iterations : int, optional
            Count of the iteration of the map, by default ``1``.

        Returns
        -------
        numpy.ndarray | float
            The image of y0 under the map.
        """
        slv = solve_poincare_map(
            y0,
            self.all_modes,
            self.trans,
            self.initial_mode,
            calc_jac=False,
            calc_hes=False,
            params=params,
            map_count=iterations,
            **self.options,
        )
        return slv.y


def solve_poincare_map(
    y0: Y,
    all_modes: tuple[Mode, ...],
    trans: dict[str, str | list[str]],
    initial_mode: str,
    calc_jac: bool = True,
    calc_hes: bool = False,
    params: P | None = None,
    rtol=1e-6,
    map_count=1,
) -> SolveIvbmpResult[Y]:
    """Solve the initial value and boundary modes problem of the hybrid dynamical system

    Solve the initial value and boundary modes problem of the hybrid dynamical system

    Parameters
    ----------
    y0 : numpy.ndarray or float
        Initial value y0.
    all_modes : tuple of Modes
        Set of all modes.
    trans : dict
        Transition function that maps from ``current mode`` to ``next mode``.
    initial_mode : str
        Name of the initial mode.
    end_mode : Mode or None, optional
        Name of the end mode, by default ``None``. If ``None``, the end mode in the method is the same as ``initial_mode``.
    calc_jac : bool, optional
        Flag to calculate the Jacobian matrix, by default ``True``.
    calc_hes : bool, optional
        Flag to calculate the Hessian tensor, by default ``True``.
    params : Parameter, optional
        Parameter to pass to ``fun`` in all `mode`, by default ``None``.
    rtol : float, optional
        Relative torelance to pass to ``solve_ivp``, by default ``1e-6``.
    map_count : int, optional
        Count of maps, by default ``1``.

    Returns
    -------
    SolveIvbmpResult

    """
    y0in, jac, eigs, eigv, hes, trans_history = _exec_calculation(
        y0,
        map_count,
        calc_jac,
        calc_hes,
        rtol,
        trans,
        all_modes,
        initial_mode,
        initial_mode,
        params,
    )

    if isinstance(y0, numpy.ndarray) and y0.size == 1:
        y0in = numpy.array(y0in)
        eigs = numpy.array(eigs)
    if not is_type_of(y0in, type(y0)):
        raise TypeError(y0in, type(y0in), y0, type(y0))
    if not is_type_of(eigs, type(y0)) and eigs is not None:
        raise TypeError(eigs, type(eigs), y0, type(y0))

    return SolveIvbmpResult[Y](y0in, trans_history, jac, eigs, eigv, hes)


def _exec_calculation(
    y0: Y,
    map_count: int,
    calc_jac: bool,
    calc_hes: bool,
    rtol: float,
    trans: dict[str, str | list[str]],
    all_modes: tuple[Mode, ...],
    initial_mode: str,
    end_mode: str | None,
    params: P | None,
):
    mdi = [m.name for m in all_modes]
    trans_history: list[str] = []
    try:
        _ = mdi.index(initial_mode)
    except KeyError as e:
        raise AllModesKeyError(initial_mode) from e
    current_mode: Mode = all_modes[mdi.index(initial_mode)]
    trans_history.append(initial_mode)

    if end_mode is None:
        end_mode = initial_mode
    else:
        try:
            _ = mdi.index(end_mode)
        except KeyError as e:
            raise AllModesKeyError(end_mode)

    jac = None
    hes = None
    count = 0

    y0in = y0 if isinstance(y0, float) else y0.copy()

    for _ in range(map_count):
        while 1:
            result = current_mode.step(
                y0in, params=params, calc_jac=calc_jac, calc_hes=calc_hes, rtol=rtol
            )
            if result.status == 0:
                raise NextModeNotFoundError

            y0in = result.y
            if not isinstance(y0in, numpy.ndarray) or y0in.size == 1:
                y0in = float(y0in)

            if calc_jac:
                if result.jac is not None:
                    if jac is None:
                        jac = numpy.eye(current_mode.fun.dom_dim)
                    jacn = numpy.array(result.jac)
                    if calc_hes:
                        if result.hes is not None:
                            if hes is None:
                                hes = numpy.zeros(
                                    (
                                        current_mode.fun.dom_dim,
                                        current_mode.fun.dom_dim,
                                        current_mode.fun.dom_dim,
                                    )
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

            try:
                current_mode = all_modes[mdi.index(next)]
            except KeyError as e:
                raise AllModesKeyError(next)
            trans_history.append(next)

            count += 1
            if current_mode.name == end_mode:
                break

    eigs = None
    eigv = None
    if jac is not None:
        if jac.size == 1:
            jac = float(jac)
            eigs = jac
        else:
            jac = numpy.squeeze(jac)
            try:  # If Jac is square
                eigs, eigv = numpy.linalg.eig(jac)
            except:
                pass
    if hes is not None:
        if isinstance(hes, float) or hes.size == 1:
            hes = float(hes)
        else:
            hes = numpy.squeeze(hes)

    return (y0in, jac, eigs, eigv, hes, trans_history)
