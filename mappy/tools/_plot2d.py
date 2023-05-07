from typing import Callable, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, KeyEvent

from numpy import ndarray

from ..typing import P
from ..fundamentals import ModeSol


class ModeColor:
    def __init__(self, color: str = "black", alpha: float = 0.3):
        self.color = color
        self.alpha = alpha


class Plot2dConfig:
    only_map: bool = False
    figsize: tuple[int, int] = (6, 6)
    xlabel: str = "x"
    ylabel: str = "y"
    xrange: tuple[float, float] = (-3, 3)
    yrange: tuple[float, float] = (-3, 3)
    xkey: int = 0
    ykey: int = 1
    linewidth: float = 1
    markersize: float = 3
    param_keys: list[str] = []  # For parameter control
    param_idx: int = 0  # For parameter control
    param_step: float = 1e-1  # For parameter control
    max_plots: int = 64

    def __init__(
        self,
        traj_color: dict[str, ModeColor] = {},
        point_color: dict[str, ModeColor] = {},
        mouse_point_color: str = "blue",
        mouse_point_alpha: float = 1,
        **kwargs,
    ):
        self.traj_color = traj_color | {"_default": ModeColor()}
        self.point_color = point_color | {"_default": ModeColor("red")}
        self.mouse_point_color = mouse_point_color
        self.mouse_point_alpha = mouse_point_alpha

        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
            else:
                raise AttributeError(f"Unknown attribute: {k}")
        self.isRunning = True


def plot2d(
    solver: Callable[[ndarray, str, P], list[ModeSol]],
    y0: ndarray,
    m0: str,
    params: P,
    config: dict[str, Any] = {},
):
    from matplotlib.animation import FuncAnimation
    from matplotlib import pyplot

    # Matplotlib initialization
    _input = [y0.copy(), m0, params.copy()]
    fig, ax, cfg = init_plot2d(_input[0], _input[2], **config)

    def update(_: int):
        sol = solver(*_input)
        if sol[-1].mtype == "D":
            _input[0][:] = sol[-1].sol
        else:
            _input[0][:] = sol[-1].sol[:, -1]
        _input[1] = sol[-1].m1

        _pc = cfg.point_color["_default"]

        for s in sol:
            if s.mtype == "C":
                if s.m0 not in cfg.traj_color:
                    _lc = cfg.traj_color["_default"]
                else:
                    _lc = cfg.traj_color[s.m0]

                if s.m0 in cfg.point_color:
                    _pc = cfg.point_color[s.m0]

                ax.plot(
                    s.sol[cfg.xkey, :],
                    s.sol[cfg.ykey, :],
                    "-",
                    linewidth=cfg.linewidth,
                    **_lc.__dict__,
                )

        ax.plot(
            _input[0][cfg.xkey],
            _input[0][cfg.ykey],
            "o",
            markersize=cfg.markersize,
            **_pc.__dict__,
        )

    _ = FuncAnimation(fig, update, interval=1, repeat=False, cache_frame_data=False)

    pyplot.show()


def init_plot2d(y0: ndarray, params: P, **kwargs) -> tuple[Figure, Axes, Plot2dConfig]:
    from matplotlib import pyplot as plt, rcParams

    _check_x0_shape(y0)

    cfg = Plot2dConfig(**kwargs)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    rcParams["keymap.fullscreen"].remove("f")

    def reset():
        for line in ax.lines:
            line.remove()

    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: _on_key_pressed(event, ax, cfg, params, reset),
    )
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: _on_click(event, ax, cfg, y0),
    )
    draw_axes2d(ax, cfg)

    return (fig, ax, cfg)


def draw_axes2d(ax: Axes, config: Plot2dConfig):
    ax.set_xlim(config.xrange)
    ax.set_ylim(config.yrange)  # type: ignore
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.grid(c="gainsboro", zorder=9)


def _on_key_pressed(
    event: KeyEvent, ax: Axes, config: Plot2dConfig, params: P, on_reset: Callable
):
    match event.key:
        case "q":
            config.isRunning = False
        case " " | "e" | "f":
            if event.key == "f":
                config.only_map = not config.only_map
            on_reset()
            draw_axes2d(ax, config)
        case "p":  # For parameter control
            config.param_idx = (config.param_idx + 1) % len(config.param_keys)
            print(f"changable parameter: {config.param_keys[config.param_idx]}")
        case "up" | "down":  # For parameter control
            step = config.param_step * (-1 if event.key == "down" else 1)
            params[config.param_keys[config.param_idx]] = round(
                params[config.param_keys[config.param_idx]] + step, 10
            )
            print({k: params[k] for k in config.param_keys})


def _on_click(
    event: MouseEvent,
    ax: Axes,
    cfg: Plot2dConfig,
    x0: ndarray,
):
    if event.xdata == None or event.ydata == None:
        return
    _check_x0_shape(x0)

    x0[cfg.xkey] = event.xdata
    x0[cfg.ykey] = event.ydata

    ax.plot(
        x0[cfg.xkey],
        x0[cfg.ykey],
        "o",
        color=cfg.mouse_point_color,
        alpha=cfg.mouse_point_alpha,
        markersize=cfg.markersize,
    )

    print(x0)
    return


def _check_x0_shape(x0: ndarray, ndim: int = 1):
    if x0.ndim != ndim:
        raise ValueError("x0 must be 1d array")
    if x0.shape[0] < 2:
        raise ValueError("x0 must have at least 2 elements")
