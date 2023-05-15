from typing import Callable, Any, Literal
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, KeyEvent

from numpy import ndarray

from ..typing import Y, P
from ..fundamentals import (
    Sol,
    ModeSol,
    ModeTraj,
    convert_y_ndarray,
    Diffeomorphism,
    PoincareMap,
)


class ModeColor:
    def __init__(self, color: str = "black", alpha: float = 0.3):
        self.color = color
        self.alpha = alpha


class Plot2dConfig:
    def __init__(
        self,
        only_map: bool = False,
        figsize: tuple[int, int] = (6, 6),
        xlabel: str = "x",
        ylabel: str = "y",
        xrange: tuple[float, float] = (-3, 3),
        yrange: tuple[float, float] = (-3, 3),
        xkey: int = 0,
        ykey: int = 1,
        linewidth: float = 1,
        markersize: float = 3,
        param_keys: list[str] = [],  # For parameter control
        param_idx: int = 0,  # For parameter control
        param_step: float = 1e-1,  # For parameter control
        max_plots: int = 64,
        float_mouse_xy: Literal["x", "y"] = "x",
        traj_color: dict[str, ModeColor] = {},
        point_color: dict[str, ModeColor] = {},
        mouse_point_color: str = "blue",
        mouse_point_alpha: float = 1,
    ):
        self.only_map = only_map
        self.figsize = figsize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xrange = xrange
        self.yrange = yrange
        self.xkey = xkey
        self.ykey = ykey
        self.linewidth = linewidth
        self.markersize = markersize
        self.param_keys = param_keys
        self.param_idx = param_idx
        self.param_step = param_step
        self.max_plots = max_plots
        self.float_mouse_xy = float_mouse_xy
        self.traj_color = traj_color | {"_default": ModeColor()}
        self.point_color = point_color | {"_default": ModeColor("red")}
        self.mouse_point_color = mouse_point_color
        self.mouse_point_alpha = mouse_point_alpha


class PlotStatus:
    def __init__(
        self,
        paused: bool = False,
        clicked: bool = False,
        key_pressed: bool = False,
    ) -> None:
        self.paused = paused
        self.clicked = clicked
        self.key_pressed = key_pressed


def plot2d(
    dif: Diffeomorphism,
    y0: Y,
    params: P,
    config: Plot2dConfig = Plot2dConfig(),
):
    f = lambda y, _, p: dif.traj(y, p)
    _plot2d(f, y0, dif.dm.name, params, config)


def mplot2d(
    pmap: PoincareMap,
    y0: Y,
    m0: str,
    params: P,
    m1: str | list[str] | None = None,
    config: Plot2dConfig = Plot2dConfig(),
):
    f = lambda y, m, p: pmap.traj(y, m, m1, p)
    _plot2d(f, y0, m0, params, config)


def _plot2d(
    solver: Callable[[ndarray, str, P], Sol | None],
    y0: Y,
    m0: str,
    params: P,
    config: Plot2dConfig = Plot2dConfig(),
):
    from matplotlib.animation import FuncAnimation
    from matplotlib import pyplot

    # Matplotlib initialization
    _input = [convert_y_ndarray(y0), m0, params.copy()]
    fig, ax, status = init_plot2d(_input[0], _input[2], config)

    def update(_: int):
        sol = solver(*_input)
        if sol is None:
            return

        if isinstance(sol.y1, float):
            raise TypeError("Invalid solution type: float")

        if not status.clicked:
            _input[0][:] = sol.y1
            if isinstance(sol, ModeSol):
                _input[1] = sol.m1

        if len(sol.trajs) > 0:
            _pc = config.point_color["_default"]

            for s in sol.trajs:
                _lc = config.traj_color["_default"]
                if isinstance(s, ModeTraj):
                    if s.m0 in config.traj_color:
                        _lc = config.traj_color[s.m0]

                    if s.m0 in config.point_color:
                        _pc = config.point_color[s.m0]

                if len(ax.lines) == 0 or status.clicked:
                    ax.plot(
                        s.sol[config.xkey, 0],
                        s.sol[config.ykey, 0],
                        ".",
                        markersize=config.markersize,
                        color=config.mouse_point_color,
                        alpha=config.mouse_point_alpha,
                    )
                    status.clicked = False

                if not config.only_map:
                    ax.plot(
                        s.sol[config.xkey, :],
                        s.sol[config.ykey, :],
                        "-",
                        linewidth=config.linewidth,
                        **_lc.__dict__,
                    )

            ax.plot(
                sol.trajs[-1].sol[config.xkey, -1],
                sol.trajs[-1].sol[config.ykey, -1],
                ".",
                markersize=config.markersize,
                **_pc.__dict__,
            )

    _ = FuncAnimation(fig, update, interval=1, repeat=False, cache_frame_data=False)

    pyplot.show()


def init_plot2d(
    y0: ndarray,
    params: P,
    cfg: Plot2dConfig,
) -> tuple[Figure, Axes, PlotStatus]:
    from matplotlib import pyplot as plt, rcParams

    status = PlotStatus()

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
        lambda event: _on_click(event, cfg, status, y0),
    )
    draw_axes2d(ax, cfg)

    return (fig, ax, status)


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
    cfg: Plot2dConfig,
    status: PlotStatus,
    x0: ndarray,
):
    if event.xdata == None or event.ydata == None:
        return

    if x0.size == 1:
        if cfg.float_mouse_xy == "x":
            x0[0] = event.xdata
        else:
            x0[0] = event.ydata
    else:
        x0[cfg.xkey] = event.xdata
        x0[cfg.ykey] = event.ydata

    print(x0)
    status.clicked = True
    return
