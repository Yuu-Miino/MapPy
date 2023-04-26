import numpy as np
from mappy import ContinuousMode as CM, DiscreteMode as DM, PoincareMap
from mappy.root import *
from mappy.tools import plot2d


## Izhikevich neuron model
@CM.function(dimension=2, param_keys=["a", "b", "I"])
def izhikevich(y: np.ndarray, param: dict | None) -> np.ndarray:
    v, u = y
    if param is None:
        raise ValueError
    a = param["a"]
    b = param["b"]
    I = param["I"]

    ret = np.array([0.04 * (v**2) + 5.0 * v + 140.0 - u + I, a * (b * v - u)])
    return ret


## Firing border
@CM.border(direction=1)
def fire_border(y, _):
    return y[0] - 30.0


## Firing jump
@DM.function(domain_dimension=2, codomain_dimension=2, param_keys=["c", "d"])
def jump(y, param):
    C = np.array([-30 + param["c"], param["d"]])
    return y + C


## Dimension conversion (p: 1 -> 2, pinv: 2 -> 1)
@DM.function(domain_dimension=1, codomain_dimension=2, param_keys=["c"])
def p(y, param):
    return np.array([0, 1]) * y + np.array([param["c"], 0])


@DM.function(domain_dimension=2, codomain_dimension=1)
def pinv(y, _):
    return np.array([0, 1]) @ y


## Main
def main():
    y0 = -1.71591635
    param = {"a": 0.2, "b": 0.2, "c": -50.0, "d": 2.0, "I": 10.0}

    all_modes = (
        DM("m0", p),
        CM("m1", izhikevich, borders=[fire_border]),
        DM("m2", jump),
        DM("m3", pinv),
    )

    transitions = {"m0": "m1", "m1": ["m2"], "m2": "m3", "m3": "m0"}

    pmap = PoincareMap(all_modes, transitions, calc_jac=True, calc_hes=True)

    f = lambda x, m, p: pmap.traj(x, m, params=p)
    config = {
        "xrange": (-80, 30),
        "yrange": (-10, 2),
        "param_keys": ["a", "I"],
        "param_idx": 0,
    }

    plot2d(
        f,
        np.array([param["c"], y0]),
        "m1",
        param,
        config=config,
    )

    """ print(pmap.image_detail(y0, "m0", params=param))

    res = trace_cycle(pmap, y0, param, "I", 4.5, show_progress=True)

    y1, p1 = res[-1][0:2]

    ret = trace_local_bf(
        poincare_map=pmap,
        y0=y1,
        params=p1,
        bf_param_idx="I",
        theta=3.14,
        cnt_param_idx="a",
        end_val=0.3,
        show_progress=True,
    )
    print(len(ret)) """


if __name__ == "__main__":
    main()
