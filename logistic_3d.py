import numpy as np
from mappy import DiscreteMode as DM, Diffeomorphism
from mappy.root import find_cycle, trace_cycle, find_local_bf
from mappy.tools import plot2d, Plot2dConfig, ColorAlpha


@DM.function(3, 3, ["alpha", "beta", "gamma"])
def logistic_3d(iny, param):
    x, y, z = iny
    alpha = param["alpha"]
    beta = param["beta"]
    gamma = param["gamma"]

    return np.array(
        [
            alpha * x * (1 - x) + beta * (y**2) * x + gamma * (z**3),
            alpha * y * (1 - y) + beta * (z**2) * y + gamma * (x**3),
            alpha * z * (1 - z) + beta * (x**2) * z + gamma * (y**3),
        ]
    )


if __name__ == "__main__":
    vecx = np.array([0.75, 0.72, 0.74])
    param = {"alpha": 3.6324, "beta": 0.0193, "gamma": 0.0146}

    dmap = Diffeomorphism("logistic_3d", logistic_3d)

    mode = "find"
    mode2 = "bf"

    if mode == "find":
        ret = find_cycle(dmap, vecx, param)
        if ret.y is None:
            raise RuntimeError("Failed to find a cycle.")

        tr = trace_cycle(
            dmap,
            ret.y,
            param,
            cnt_param_key="alpha",
            end_val=2.95,
            show_progress=True,
        )
        if tr is None:
            raise RuntimeError("Failed to trace a cycle.")

        if mode2 == "graph":
            alphas = [t[1]["alpha"] for t in tr if t[2] is not None]
            eigs = [np.abs(t[2]["eigvals"]) for t in tr if t[2] is not None]

            import matplotlib.pyplot as plt

            plt.grid()
            plt.plot(alphas, eigs)
            plt.show()

        elif mode2 == "bf":
            y1 = tr[-1][0]
            p1 = tr[-1][1]
            e1 = tr[-1][2]["eigvals"]

            print(y1, p1, e1)

            print(find_local_bf(dmap, y1, p1, "alpha", np.pi, 1))

    elif mode == "traj":
        config = Plot2dConfig(
            xrange=(0, 1.2),
            yrange=(0, 1.2),
            ykey=2,
            param_keys=["alpha", "beta", "gamma"],
            param_idx=0,
            traj_color=ColorAlpha("orange"),
            point_color=ColorAlpha("teal"),
            only_map=True,
        )
        plot2d(dmap, vecx, param, config)
