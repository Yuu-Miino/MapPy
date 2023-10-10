import numpy as np
from mappy import ContinuousMode as CM, DiscreteMode as DM, PoincareMap
from mappy.root import *
from sympy import cos
from mappy.tools import mplot2d, Plot2dConfig


@CM.function(dimension=3, param_keys=["k", "B0", "B"])
def duffing(iny, param):
    x, y, t = iny
    k = param["k"]
    B = param["B"]
    B0 = param["B0"]

    return np.array([y, -k * y - x**3 + B0 + B * cos(t), 1])


@CM.border(1)
def poin_sec(iny, _):
    return iny[2] - 2 * np.pi


@DM.function(3, 3)
def t_reset(iny, _):
    return iny - np.array([0, 0, 2 * np.pi])


@DM.function(2, 3)
def p(iny, _):
    return np.append(iny, 0)


@DM.function(3, 2)
def pinv(iny, _):
    return iny[0:2]


def main():
    y0 = np.array([0.232147751858, 0.075591687407])
    param = {"k": 0.2, "B0": 0.1, "B": 0.1}

    all_modes = (
        CM("duffing", duffing, borders=[poin_sec]),
        DM("t_reset", t_reset),
        DM("p", p),
        DM("pinv", pinv),
    )

    trans = {"p": "duffing", "duffing": ["t_reset"], "t_reset": "pinv", "pinv": "p"}

    pmap = PoincareMap(all_modes, trans, "p")

    config = Plot2dConfig(
        xrange=(-1.5, 1.5),
        yrange=(-1.5, 1.5),
        param_keys=["B0", "B"],
        param_idx=0,
    )

    run_mode = "single_bf"

    match run_mode:
        case "traj":
            mplot2d(pmap, y0, "p", param, config=config)
        case "fix_trace":
            ret = trace_cycle(
                diff=pmap,
                y0=y0,
                m0="p",
                params=param,
                period=1,
                cnt_param_key="B",
                end_val=0.375,
                show_progress=True,
                resolution=50,
            )
            pxye = np.array(
                [[r[1]["B0"], r[1]["B"], *r[0], *r[2]["eigvals"]] for r in ret],
                dtype=np.floating,
            )
            np.savetxt("duffing_fix_trace.csv", pxye, delimiter=",")
        case "bf_Trace":
            ret_lbf = trace_local_bf(
                diff=pmap,
                y0=y0,
                m0="p",
                params=param,
                bf_param_key="B0",
                theta=np.pi,
                cnt_param_key="B",
                end_val=0.35,
                period=1,
                resolution=100,
                show_progress=True,
            )
            print(len(ret_lbf))
        case "1dim_bf":
            resolution = 1000
            Brange = np.linspace(0.1, 0.375, resolution)
            maps = 100
            with open("duffing_1dim_bf.csv", "w") as f:
                c = 0
                for B in Brange:
                    param["B"] = B
                    for i in range(maps):
                        y0 = pmap.image(y0=y0, m0="p", params=param)
                        f.write(f"{param['B0']},{param['B']},{y0[0]},{y0[1]}\n")
                        print(
                            "\t",
                            f"[{c:04d}/{resolution:04d} | {i:03d}]",
                            "B0:",
                            param["B0"],
                            "B:",
                            param["B"],
                            "y0:",
                            y0,
                            end="\r",
                        )
                    f.write("\n")
                    print()
                    c += 1

        case "single_bf":
            y0 = np.array([-3.115916368789392599e-01, 1.422745846218321231e-01])
            param["B"] = 3.6938e-01

            ret = find_local_bf(
                pmap, y0, m0="p", params=param, param_key="B", theta=np.pi
            )
            print(ret)
        case _:
            pass


if __name__ == "__main__":
    main()
