import numpy as np
from sympy import sin

from mappy import ContinuousMode as CM, DiscreteMode as DM, PoincareMap
from mappy.root import *
from mappy.plot import mplot2d, Plot2dConfig, ColorAlpha


@CM.function(3, ["B0", "omega", "k1", "B1", "ALPHA"])
def const_bvp_ON(vec_y, param):
    x, y, z = vec_y

    k1 = param["k1"]
    B0 = param["B0"]
    B1 = param["B1"]
    omega = param["omega"]

    return np.array(
        [
            0,
            -x - k1 * y + B0 + B1 * sin(omega * z),
            1,
        ]
    )


@CM.border(-1)
def const_bvp_ON_border(vec_y, param):
    ALPHA = param["ALPHA"]
    Gx = -ALPHA + ALPHA**3
    return vec_y[1] - Gx


@CM.function(3, ["B0", "omega", "EPSILON", "k1", "B1", "ALPHA"])
def const_bvp_OFF(vec_y, param):
    x, y, z = vec_y

    EPSILON = param["EPSILON"]
    k1 = param["k1"]
    B0 = param["B0"]
    B1 = param["B1"]
    omega = param["omega"]

    Gx = -x + x**3

    return np.array(
        [
            (y - Gx) / EPSILON,
            -x - k1 * y + B0 + B1 * sin(omega * z),
            1,
        ]
    )


@CM.border()
def const_bvp_OFF_border(vec_y, param):
    return vec_y[0] - param["ALPHA"]


@CM.border()
def poincare_section(vec_y, param):
    return vec_y[2] - 2 * np.pi / param["omega"]


@DM.function(3, 3, ["omega"])
def p_reset(vec_y, param):
    return vec_y - np.array([0, 0, 2 * np.pi / param["omega"]])


@DM.function(3, 2)
def p_reduce(vec_y, _):
    return vec_y[0:2]


@DM.function(2, 3)
def p_expand(vec_y, _):
    return np.array([vec_y[0], vec_y[1], 0])


def main():
    all_modes = (
        CM("ON", const_bvp_ON, borders=[const_bvp_ON_border, poincare_section]),
        CM("OFF", const_bvp_OFF, borders=[const_bvp_OFF_border, poincare_section]),
        DM("ON_Reset", p_reset),
        DM("ON_Pr", p_reduce),
        DM("ON_Pe", p_expand),
        DM("OFF_Reset", p_reset),
        DM("OFF_Pr", p_reduce),
        DM("OFF_Pe", p_expand),
    )

    transitions = {
        "ON": ["OFF", "ON_Reset"],
        "OFF": ["ON", "OFF_Reset"],
        "OFF_Reset": "OFF_Pr",
        "OFF_Pr": "OFF_Pe",
        "OFF_Pe": "OFF",
        "ON_Reset": "ON_Pr",
        "ON_Pr": "ON_Pe",
        "ON_Pe": "ON",
    }

    pmap = PoincareMap(all_modes, transitions, ["ON_Pe", "OFF_Pe"], rtol=1e-6)

    y0 = [0.7918850482271097, -0.3133095637621468]
    m0 = "OFF_Pe"
    params = {
        "EPSILON": 0.1,
        "k1": 0.9,
        "B0": 0.207,
        "B1": 1.0e-2,
        "omega": 0.458,
        "ALPHA": 0.8,
    }

    mode = "analyze"

    if mode == "traj":
        mplot2d(
            pmap,
            y0,
            m0,
            params,
            Plot2dConfig(
                xlabel="x",
                ylabel="y",
                xrange=(-1.2, 1.0),
                yrange=(-0.6, 1.0),
                traj_color={"ON": ColorAlpha("orange"), "OFF": ColorAlpha("teal")},
                param_keys=["omega", "B1"],
                param_idx=0,
                param_step=1e-4,
                markersize=5,
            ),
        )
    elif mode == "analyze":
        # y0 = pmap.image(y0, m0, params, 100)
        print(y0)

        mode2 = "bf2"

        if mode2 == "plot":
            ret = find_cycle(pmap, y0, params, m0=m0)
            print(ret)

            y1 = ret.y
            if y1 is None:
                raise ValueError
            ret = trace_cycle(
                pmap,
                y1,
                params,
                "omega",
                0.45633,
                m0=m0,
                show_progress=True,
                resolution=50,
            )

            data_x = [r[1]["omega"] for r in ret]
            date_y = [r[0][1] for r in ret]
            date_eg = [None if r[2] is None else r[2]["eigvals"] for r in ret]

            import matplotlib.pyplot as plt

            ax1 = plt.subplot(211)
            ax1.plot(data_x, date_y)
            ax1.plot(data_x, [-0.8 + 0.8**3] * len(data_x))
            ax2 = plt.subplot(212)
            ax2.plot(data_x, date_eg)
            plt.show()

            print(ret[-1])
        elif mode2 == "bf":
            ybf = np.array([0.7475852485542821, -0.35897748827647435])
            params["B1"] = 0.011818181818181814
            params["omega"] = 0.4503629577229586
            print(ybf)
            period = 1

            OK = False

            if not OK:
                sol = pmap.traj(ybf, m0, params, 1)
                if sol is None:
                    raise ValueError

                import matplotlib.pyplot as plt

                ax1 = plt.subplot(211)
                ax1.grid()
                [ax1.plot(t.sol[2, :], t.sol[0, :], "-") for t in sol.trajs]
                ax2 = plt.subplot(212)
                ax2.grid()
                [ax2.plot(t.sol[2, :], t.sol[1, :], "-") for t in sol.trajs]
                plt.show()

                np.savetxt(
                    "traj_gz.dat",
                    np.hstack([t.sol for t in sol.trajs]).T,
                    delimiter=" ",
                )

            else:
                ret = trace_local_bf(
                    pmap, ybf, params, "omega", 0, "B1", 1.4e-2, 100, period, True, m0
                )
                dat = [
                    [
                        *r[1].values(),
                        "OFF" if m0 == "OFF_Pe" else "ON",
                        period,
                        *r[0][0],
                        r[0][1],
                    ]
                    for r in ret
                ]
                import csv

                with open("data.csv", "w") as f:
                    writer = csv.writer(f, delimiter=" ")
                    writer.writerows(dat)
        elif mode2 == "bf2":
            ybf = np.array([0.786497791531, -0.321391109487])
            params["omega"] = 0.45633
            period = 1

            ret = trace_local_bf(
                pmap, ybf, params, "omega", np.pi, "B1", 1.4e-2, 100, period, True, m0
            )
            dat = [
                [
                    *r[1].values(),
                    "OFF" if m0 == "OFF_Pe" else "ON",
                    period,
                    *r[0][0],
                    r[0][1],
                ]
                for r in ret
            ]
            import csv

            with open("data2.csv", "w") as f:
                writer = csv.writer(f, delimiter=" ")
                writer.writerows(dat)

    else:
        pass


if __name__ == "__main__":
    main()
