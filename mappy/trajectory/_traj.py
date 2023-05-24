from typing import Literal, Sequence
import numpy

from ..typing import Y


class Traj:
    """
    Represents a trajectory.

    Parameters
    ----------
    sol : numpy.ndarray
        Array representing the trajectory.

    See Also
    --------
    ModeTraj : Represents a mode trajectory.

    """

    def __init__(self, sol: numpy.ndarray) -> None:
        self.sol = sol


class ModeTraj(Traj):
    """
    Represents a mode trajectory.

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
    Traj : Represents a trajectory.
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
    Represents a solution.

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
    mappy.typing.Y : Type alias for ``y0`` and ``y1``.

    """

    def __init__(self, y0: Y, y1: Y, trajs: Sequence[Traj]) -> None:
        self.y0 = y0
        self.y1 = y1
        self.trajs = trajs


class ModeSol(Sol):
    """
    Represents a solution with mode information.

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
    trajs : list[Traj]
        List of trajectories.

    See Also
    --------
    Sol : Represents a solution.
    mappy.ModeTraj : Represents a mode trajectory.
    mappy.typing.Y : Type alias for ``y0`` and ``y1``.

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
