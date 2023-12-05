import numpy as np
from mappy import DiscreteMode as DM, Diffeomorphism
from mappy.tools import plot2d, Plot2dConfig, ColorAlpha


@DM.function(2, 2, ["a", "b"])
def henon(iny, param):
    x, y = iny
    a = param["a"]
    b = param["b"]

    return np.array([1 - a * x**2 + y, b * x])


if __name__ == "__main__":
    vecx = np.array([-1.06770401965, 0.368194183333])
    param = {"a": 1.42, "b": 0.3}

    diff = Diffeomorphism("henon", henon)

    config = Plot2dConfig(
        xrange=(-1.5, 1.5),
        yrange=(-0.5, 0.5),
        param_keys=["a", "b"],
        param_idx=0,
        traj_color=ColorAlpha("orange"),
        point_color=ColorAlpha("teal"),
        only_map=True,
    )

    MODE = "MLE"

    if MODE == "plot":
        plot2d(diff, vecx, param, config)
    elif MODE == "MLE":
        N = 10000
        RES = 500
        dat = np.empty((RES, 2))
        _x0 = vecx

        for k, p in enumerate(np.linspace(1, 1.42, RES)):
            param["a"] = p
            unit = np.array([1, 0], dtype=np.float64)
            mle = 0

            for i in range(N):
                res = diff.image_detail(_x0, param, calc_hes=False)
                _x0 = res.y
                jac = res.jac
                if isinstance(jac, np.ndarray):
                    unit = jac @ unit
                    norm = np.linalg.norm(unit)
                    mle += np.log(norm)
                    unit /= norm

            print(k, p, mle / N)
            dat[k, :] = [p, mle / N]

        np.savetxt("henon_mle.csv", dat, delimiter=",")
    elif MODE == "bf1":
        N = 2000
        TRANS = 1500
        RES = 2000
        dat = np.empty((N - TRANS, RES, 3))
        _x0 = vecx

        for k, p in enumerate(np.linspace(1, 1.42, RES)):
            param["a"] = p

            for i in range(N):
                _x0 = diff.image(_x0, param)
                if i >= TRANS:
                    dat[i - TRANS, k, :] = np.hstack([p, _x0])

            print(k, p)

        with open("henon_bf1.csv", "w") as f:
            for k in range(RES):
                for i in range(N - TRANS):
                    f.write(
                        "{}, {}, {}\n".format(dat[i, k, 0], dat[i, k, 1], dat[i, k, 2])
                    )
                f.write("\n")
