from typing import Literal
from ..typing import is_type_of


class MatplotConfig:
    only_map: bool = False
    figsize: tuple[int, int] = (6, 6)
    xlabel: str = "x"
    ylabel: str = "y"
    xrange: tuple[int, int] = (-3, 3)
    yrange: tuple[int, int] = (-3, 3)
    xkey: int = 0
    ykey: int = 1
    linewidth: float = 1
    pointsize: float = 3
    alpha: float = 0.3
    traj_color: str = "black"
    point_color: str = "red"
    param_keys: list[str] = []  # For parameter control
    param_idx: int = 0  # For parameter control
    param_step: float = 1e-2  # For parameter control
    max_plots: int = 64

    def __init__(self, **kwargs):
        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
        self.isRunning = True


def init_axes(**kwargs):
    from matplotlib import pyplot as plt

    cfg = MatplotConfig(**kwargs)

    fig = plt.figure(figsize=cfg.figsize)
    option = 111
    if not is_type_of(option, Literal["3d"]):
        raise TypeError

    ax = plt.subplot()
