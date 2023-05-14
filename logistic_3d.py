import numpy as np
from mappy import DiscreteMode as DM, PoincareMap
from mappy.tools import plot2d, Plot2dConfig, ModeColor


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
    vecx = np.array([0.4212, 0.1436, 0.7108])
    param = {"alpha": 3.6324, "beta": 0.0193, "gamma": 0.0146}

    all_modes = (DM("logistic_3d", logistic_3d, True),)
    trans = {"logistic_3d": "logistic_3d"}

    pmap = PoincareMap(all_modes, trans, True, True)

    print(pmap.traj(vecx, "logistic_3d", params=param))

    f = lambda x, m, p: pmap.traj(x, m, params=p)

    config = Plot2dConfig(
        xrange=(0, 1.2),
        yrange=(0, 1.2),
        param_keys=["alpha", "beta", "gamma"],
        param_idx=0,
        traj_color={"logistic_3d": ModeColor("orange")},
        only_map=True,
    )

    plot2d(f, vecx, "logistic_3d", param, config)
