from typing import Callable, Any
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, KeyEvent
from matplotlib.lines import Line2D

from numpy import ndarray

from ..typing import P
from ..fundamentals import ModeSol


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
    pointsize: float = 3
    alpha: float = 0.3
    traj_color: str = "black"
    point_color: str = "red"
    mouse_point_color: str = "blue"
    param_keys: list[str] = []  # For parameter control
    param_idx: int = 0  # For parameter control
    param_step: float = 1e-1  # For parameter control
    max_plots: int = 64

    def __init__(self, **kwargs):
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
    keys = ["xkey", "ykey"]
    all_sol = {k: [] for k in keys}
    all_pt = {k: [] for k in keys}

    def reset_arr():
        for k in keys:
            all_sol[k].clear()
            all_pt[k].clear()

    from matplotlib.animation import FuncAnimation
    from matplotlib import pyplot

    # Matplotlib initialization
    _y0 = y0.copy()
    _params = params.copy()
    _m0 = [m0]
    fig, cfg, ln, pt = init_plot2d(_y0, _params, reset_arr, **config)

    def update(_: int):
        sol = solver(_y0, _m0[0], _params)
        if sol[-1].mtype == "D":
            _y0[:] = sol[-1].sol
        else:
            _y0[:] = sol[-1].sol[:, -1]
        _m0[0] = sol[-1].m1

        for k in keys:
            [
                all_sol[k].extend(s.sol[getattr(cfg, k), :])
                for s in sol
                if s.mtype == "C"
            ]
            all_pt[k].append(_y0[getattr(cfg, k)])

        ln.set_data(all_sol["xkey"], all_sol["ykey"])
        pt.set_data(all_pt["xkey"], all_pt["ykey"])

    _ = FuncAnimation(fig, update, interval=1, repeat=False, cache_frame_data=False)

    pyplot.show()


def init_plot2d(
    y0: ndarray, params: P, on_reset: Callable, **kwargs
) -> tuple[Figure, Plot2dConfig, Line2D, Line2D]:
    from matplotlib import pyplot as plt, rcParams

    _check_x0_shape(y0)

    cfg = Plot2dConfig(**kwargs)

    fig, ax = plt.subplots(figsize=cfg.figsize)
    rcParams["keymap.fullscreen"].remove("f")

    (ln,) = ax.plot(
        [],
        [],
        linewidth=cfg.linewidth,
        color=cfg.traj_color,
        ls="-",
        alpha=cfg.alpha,
    )
    (pt,) = ax.plot(
        [],
        [],
        "o",
        markersize=cfg.pointsize,
        color=cfg.point_color,
        alpha=cfg.alpha,
    )
    (mpt,) = ax.plot(
        [],
        [],
        "o",
        markersize=cfg.pointsize,
        color=cfg.mouse_point_color,
        alpha=cfg.alpha,
    )

    mpt.set_data(y0)

    old_trajs: list[Line2D] = []
    old_pts: list[Line2D] = []
    old_mpts: list[Line2D] = []

    def reset():
        on_reset()
        mpt.set_data([], [])
        for ls in (old_trajs, old_pts, old_mpts):
            for l in ls:
                l.remove()
            ls.clear()

    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: _on_key_pressed(event, ax, cfg, params, reset),
    )
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: _on_click(
            event, ax, ln, pt, mpt, cfg, y0, on_reset, old_trajs, old_pts, old_mpts
        ),
    )
    draw_axes2d(ax, cfg)

    return (fig, cfg, ln, pt)


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
    ln: Line2D,
    pt: Line2D,
    mpt: Line2D,
    cfg: Plot2dConfig,
    x0: ndarray,
    reset: Callable,
    old_trajs: list[Line2D],
    old_pts: list[Line2D],
    old_mpts: list[Line2D],
):
    if event.xdata == None or event.ydata == None:
        return
    _check_x0_shape(x0)

    x0[cfg.xkey] = event.xdata
    x0[cfg.ykey] = event.ydata

    old_ln = ln.get_data(True)
    old_pt = pt.get_data(True)
    old_mpt = mpt.get_data(True)

    (oln,) = ax.plot(
        old_ln[0],
        old_ln[1],
        linewidth=cfg.linewidth,
        color=cfg.traj_color,
        ls="-",
        alpha=cfg.alpha,
    )
    (opt,) = ax.plot(
        old_pt[0], old_pt[1], "o", markersize=cfg.pointsize, color=cfg.point_color
    )
    (ompt,) = ax.plot(
        old_mpt[0],
        old_mpt[1],
        "o",
        markersize=cfg.pointsize,
        color=cfg.mouse_point_color,
        alpha=cfg.alpha,
    )

    old_trajs.append(oln)
    old_pts.append(opt)
    old_mpts.append(ompt)

    ln.set_data([], [])
    pt.set_data([], [])
    mpt.set_data([], [])
    reset()

    mpt.set_data(
        x0[cfg.xkey],
        x0[cfg.ykey],
    )

    print(x0)
    return


def _check_x0_shape(x0: ndarray, ndim: int = 1):
    if x0.ndim != ndim:
        raise ValueError("x0 must be 1d array")
    if x0.shape[0] < 2:
        raise ValueError("x0 must have at least 2 elements")
