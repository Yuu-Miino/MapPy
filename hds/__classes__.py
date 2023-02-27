from typing import Concatenate, ParamSpec, TypeVar
from collections.abc import Callable
import numpy
from scipy.integrate import solve_ivp, OdeSolution

P = ParamSpec('P')
Y = TypeVar('Y', numpy.ndarray, float)
YF = TypeVar('YF', numpy.ndarray, float)
YJ = TypeVar('YJ', numpy.ndarray, float)
YB = TypeVar('YB', numpy.ndarray, float)

class JacDimErr (Exception):
    """Exception for the mismatch of the dimensions.

    Parameters
    ----------
    dim : int
        Specified dimension of the system domain.
    dim_to : int
        Specified dimension of the system codomain.
    len : int
        Length of the calculated result of `fun`.
    """
    def __init__(self, dom_dim: int, cod_dim : int, length: int) -> None:
        self.dom_dim = dom_dim
        self.cod_dim = cod_dim
        self.length = length
    def __str__(self) -> str:
        return (
            f'[Jacobian Matrix] Dimension not match: {self.dom_dim} + ({self.dom_dim} * {self.cod_dim}) and {self.length}. '
            'Please check `dim` and `fun` passed to the mode.'
        )

class ModeStepResult:
    """Result of `step` in `Mode`

    Parameters
    ----------
    status : int
        Response status of solve_ivp for the continuous mode.
        `0` for the discrete mode.
    y : numpy.ndarray or float
        Value of the solution after step.
    tend : float or None, optional
        Value of the time after step of the continuous-time mode, by default `None`.
    i_border: int or None, optional
        Index of the border where the trajectory arrives, by default `None`.
    jac: numpy.ndarray, float, or None, optional
        Value of the Jacobian matrix, by default `None`.
    sol: OdeSolution or None, optional
        OdeSolution instance of `solve_ivp` in the continuous-time mode, by default `None`.
    """
    def __init__(self,
        status: int,
        y: numpy.ndarray | float,
        tend: float | None = None,
        jac: numpy.ndarray | float | None = None,
        i_border: int | None = None,
        sol: OdeSolution | None = None
    ) -> None:
        self.status = status
        self.y = y
        self.i_border = i_border
        self.jac = jac
        self.sol = sol
        self.tend = tend

    def __repr__(self) -> str:
        return str(self.__dict__)

class Mode:
    """Parent Class of all modes
    """
    def __init__(self,
        fun: Callable[Concatenate[Y, P], YF],
        jac_fun: Callable[Concatenate[Y, P], YJ] | None = None,
    ) -> None:
        self.fun = fun
        self.jac_fun = jac_fun

    def __eq__(self, other):
        if not isinstance(other, Mode):
            return NotImplemented
        return id(self) == id(other)

    def step(self, y0: numpy.ndarray | float, args = None, **options) -> ModeStepResult:
        """Step to the next mode

        Parameters
        ----------
        y0 : numpy.ndarray
            The initial state to pass to `fun`.
        args : Any or None, optional
            The parameter to pass to `fun`, by default None.
        **options
            For future implementation.

        Returns
        -------
        ModeStepResult
        """

        return NotImplemented

class ContinuousMode (Mode):
    """Mode for the continuos-time dynamical system

    Parameters
    ----------
    fun : Callable
        Right-hand side of the continuous-time dynamical system. The calling signature is fun(y).
    borders: list of callables
        List of the border functions to pass to `solve_ivp` as events. The calling signature is border(y).
    jac_fun : Callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect to y, by default `None`.
    db_dy : list of callables or None, optional
        List of the derivative of the border functions with respect to the state `y`, by default `None`.
        It is necessary for the Jacobian matrix calculation.
    max_interval: float, optional
        Max interval of the time span, by default `20`.
        The function `solve_ivp` of SciPy takes `t_span = [0, max_interval]`.

    """
    next: list[Mode]
    """List of the next modes after arriving `borders`.
    The index of `next` must correspond with the index of `borders`.
    """

    def __init__(self,
        fun: Callable[Concatenate[Y, P], YF],
        borders: list[Callable[Concatenate[Y, P], float]],
        jac_fun: Callable[Concatenate[Y, P], YJ] | None = None,
        db_dy: list[Callable[Concatenate[Y, P], YB]] | None = None,
        max_interval: float = float(20),
    ) -> None:
        super().__init__(fun, jac_fun)
        self.max_interval = max_interval
        self.borders = borders
        self.db_dy = db_dy

    def __ode_jac (self, y: numpy.ndarray, ode_fun: Callable[Concatenate[Y, P], YF], jac_fun: Callable[Concatenate[Y, P], YJ], args=None) -> numpy.ndarray:
        dim = ode_fun.dom_dim
        try:
            dydy0 = y[dim:dim+(dim**2)].reshape((dim, dim), order='F')
        except ValueError:
            raise JacDimErr(dom_dim=dim, cod_dim=dim, length=len(y)) from None

        ode = lambda y, fun=ode_fun: fun(y, *args)
        ode_jac = lambda y, fun=jac_fun: fun(y, *args)

        deriv = numpy.empty(dim+(dim**2))
        deriv[0:dim] = ode(y[0:dim])
        jac = ode_jac(y[0:dim])

        deriv[dim:dim+(dim**2)] = (jac @ dydy0).flatten(order='F')
        return deriv

    @classmethod
    def function(cls, dimension: int) -> Callable:
        """Decorator for `fun` in `ContinuousTimeMode`

        Parameters
        ----------
        dimension : int
            Dimension of the state space.

        Returns
        -------
        Callable
            Decorated `fun` function compatible with `ContinousTimeMode`
        """
        def _decorator(fun: Callable[P, YF]) -> Callable[P, YF]:
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> YF:
                ret = fun(*args, **kwargs)
                return ret
            _wrapper.dom_dim = dimension
            return _wrapper
        return _decorator

    @classmethod
    def border(cls, direction: int = 1) -> Callable:
        """Decorator for the element of `borders` in `ContinuousTimeMode`

        Parameters
        ----------
        direction : int, optional
            Direction of a zero crossing, by default `1`.
            The value is directly passed to the `direction` attribute of the `event` function, which is the argument of `solve_ivp`.

        Returns
        -------
        Callable
            Decorated `border` function compatible with `ContinuousTimeMode`
        """
        def _decorator(fun: Callable[P, YF]) -> Callable[P, YF]:
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> YF:
                ret = fun(*args, **kwargs)
                return ret
            _wrapper.direction = direction
            return _wrapper
        return _decorator

    def step(self, y0: numpy.ndarray | float, args = None, **options)->ModeStepResult:
        """Step to the next mode

        Parameters
        ----------
        y0 : numpy.ndarray or float
            The initial state y0 of the system evolution.
        args : Any, optional
            Arguments to pass to `fun`, `jac_fun`, and `borders`, by default None.
        **options
            The options of `solve_ivp`.

        Returns
        -------
        ModeStepResult

        Raises
        ------
        JacDimErr
            Error of the dimension of the Jacobian matrix.
        """

        # Replace the function for ODE and borders with the compatible forms
        if self.jac_fun is None:
            ode = lambda t, y, fun = self.fun: fun(y, *args)
        else:
            ode = lambda t, y, fun = self.__ode_jac, ode_fun=self.fun, jac_fun = self.jac_fun: fun(y, ode_fun, jac_fun, args)
        borders = []
        for i, ev in enumerate(self.borders):
            evi = lambda t, y, ev=ev: ev(y, *args)
            evi.terminal  = True
            evi.direction = ev.direction
            borders.append(evi)
        devs = []
        if self.db_dy is not None:
            for i, dev in enumerate(self.db_dy):
                devi = lambda t, y, dev=dev: dev(y, *args)
                devs.append(devi)
            if len(borders) != len(devs):
                raise IndexError('Lists `db_dt` and `borders` must be the same size.')

        i_border: int | None = None
        dim = self.fun.dom_dim

        # Copy initial value
        if isinstance(y0, numpy.ndarray):
            y0in = y0.copy()
        else:
            y0in = y0

        # Append an identity matrix to the initial state if calculate Jacobian matrix
        if self.jac_fun is not None:
            y0in = numpy.append(y0in, numpy.eye(dim).flatten())

        ## Main loop: solve initial value problem
        sol = solve_ivp(ode, [0, self.max_interval], y0in, events=borders, **options)

        ## Set values to the result instance
        y1  = sol.y.T[-1][0:dim] if dim != 1 else sol.y.T[-1][0]
        jac = numpy.eye(dim) if self.jac_fun is not None else None

        if self.jac_fun is not None: # If calculate Jacobian matrix
            jac = sol.y.T[-1][dim:dim+(dim**2)].reshape((dim, dim)).T

        # border detect
        if sol.status == 1:
            # For each borders
            for i, ev in enumerate(self.borders):
                if len(sol.t_events[i]) != 0:
                    if self.jac_fun is not None:
                        dydt = ode(0, numpy.append(y1, numpy.eye(dim).flatten()))[0:dim]
                        dbdy = devs[i](0, y1)
                        jac = (numpy.eye(dim) - 1.0/numpy.dot(dbdy, dydt) * numpy.outer(dydt, dbdy)) @ jac
                    i_border = i

        # Make a response instance
        result = ModeStepResult(sol.status, y1, tend=sol.t[-1], jac=jac, i_border=i_border)
        if options.get('dense_output'):
            result.sol = sol.sol

        return result

class DiscreteMode (Mode):
    """Mode of the discrete-time dynamical system

    Parameters
    ----------
    fun : Callable
        Right-hand side of the discrete-time dynamical system. The calling signature is fun(y).
    jac_fun : Callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect to y, by default `None`.
    """
    next: Mode
    """Next modes after the mapping.
    """

    def __init__(self,
        fun: Callable[Concatenate[Y, P], YF],
        jac_fun: Callable[Concatenate[Y, P], YJ] | None = None
    ) -> None:
        super().__init__(fun, jac_fun)

    @classmethod
    def function(cls, domain_dimension: int, codomain_dimenstion: int) -> Callable:
        """Decorator for `fun` in `DiscreteTimeMode`

        Parameters
        ----------
        domain_dimension : int
            Dimension of the domain of the function `fun`.
        codomain_dimenstion : int
            Dimension of the codomain of the function `fun`.

        Returns
        -------
        Callable
            Decorated function compatible with `DiscreteTimeMode`
        """
        def _decorator(fun: Callable[P, YF]) -> Callable[P, YF]:
            def _wrapper(*args: P.args, **kwargs: P.kwargs) -> YF:
                ret = fun(*args, **kwargs)
                return ret
            _wrapper.dom_dim = domain_dimension
            _wrapper.cod_dim = codomain_dimenstion
            return _wrapper
        return _decorator

    def step(self, y0: numpy.ndarray | float, args = None, **options) -> ModeStepResult:
        """Step to the next mode

        Parameters
        ----------
        y0 : numpy.ndarray or float
            The initial state y0 of the system evolution.
        args : Any, optional
            Arguments to pass to `fun` and `jac_fun`, by default None.
        **options
            For future implementation.

        Returns
        -------
        ModeStepResult

        Raises
        ------
        JacDimErr
            Error of the dimension of the Jacobian matrix.
        """
        ## Setup
        y1  = y0 if isinstance(y0, float) else y0.copy()
        i_border: int | None = None
        jac = None

        # Convert functions into the general form
        if self.jac_fun is None:
            mapT = lambda n, y, fun = self.fun: numpy.array(fun(y, *args))
        else:
            mapT = lambda n, y, fun = self.fun, jac_fun=self.jac_fun: numpy.append(
                fun(y, *args),
                numpy.array(jac_fun(y, *args)).flatten(order='F')
            )

        ## Main part
        sol  = mapT(0, y1)

        if isinstance(sol, float):
            y1 = sol
        elif self.fun.cod_dim == 1:
            y1 = sol[0]
        else:
            y1 = sol[0:self.fun.cod_dim]

        if self.jac_fun is not None:
            try:
                jac = sol[self.fun.cod_dim:self.fun.cod_dim+(self.fun.dom_dim*self.fun.cod_dim)].reshape((self.fun.cod_dim, self.fun.dom_dim), order='F')
            except ValueError:
                raise JacDimErr(dom_dim=self.fun.dom_dim, cod_dim=self.fun.cod_dim, length=len(sol)) from None

        result = ModeStepResult(status=0, y=y1, jac=jac, i_border=i_border)

        return result

class DictVector:
    """Class for vectors accepting the dictionary input.

    Example
    -------
    >>> class Parameter(DictVector):
    ...     pass
    >>> data = {'a': 0.2, 'b': 0.1, 'c': -0.3}
    >>> p = Parameter(data)
    >>> p.a
    0.2
    >>> p.b
    0.1
    >>> p.c
    -0.3
    >>> p
    {'a': 0.2, 'b': 0.1, 'c': -0.3}
    """
    def __init__(self, states: dict[str, float] |  None = None) -> None:
        if states is not None:
            for key, val in states.items():
                setattr(self, key, val)
    def __repr__(self) -> str:
        return str(self.__dict__)
