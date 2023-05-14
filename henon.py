import numpy as np
from mappy import DiscreteMode as DM, PoincareMap
from mappy.tools import plot2d, Plot2dConfig, ModeColor


@DM.function(2, 2, ["a", "b"])
def henon(iny, param):
    x, y = iny
    a = param["a"]
    b = param["b"]

    return np.array([1 - a * x**2 + y, b * x])


if __name__ == "__main__":
    vecx = np.array([0.1, 0.1])
    param = {"a": 1.4, "b": 0.3}

    all_modes = (DM("henon", henon, True),)
    trans = {"henon": "henon"}

    pmap = PoincareMap(all_modes, trans, True, True)

    print(pmap.traj(vecx, "henon", params=param))

    f = lambda x, m, p: pmap.traj(x, m, params=p)

    config = Plot2dConfig(
        xrange=(-1.5, 1.5),
        yrange=(-0.5, 0.5),
        param_keys=["a", "b"],
        param_idx=0,
        traj_color={"henon": ModeColor("orange")},
        only_map=True,
    )

    plot2d(f, vecx, "henon", param, config)
