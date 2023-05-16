import numpy as np
from mappy import DiscreteMode as DM, Diffeomorphism
from mappy.root import find_cycle
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

    # find_cycle()

    plot2d(dmap, vecx, param, config)
