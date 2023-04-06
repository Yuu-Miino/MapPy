import numpy as np
from sympy import cos
from mappy import ContinuousMode as CM, DiscreteMode as DM, PoincareMap
from mappy.root import *

@CM.function(3, ['k', 'V0', 'V', 'omega', 'g1', 'B1'])
def alp_1 (y, param):
    v1, v2, t = y
    k = param['k']
    V0 = param['V0']
    V = param['V']
    omega = param['omega']
    g1 = param['g1']
    B1 = param['B1']

    return np.array([
        -k * v1 - v2 + V0 + V * cos(omega * t),
        v1 + (1-g1) * v2 - (v2 ** 3) / 3 + B1,
        1
    ])

@CM.function(3, ['k', 'V0', 'V', 'omega', 'g2', 'B2'])
def alp_2 (y, param):
    v1, v2, t = y
    k = param['k']
    V0 = param['V0']
    V = param['V']
    omega = param['omega']
    g2 = param['g2']
    B2 = param['B2']

    return np.array([
        -k * v1 - v2 + V0 + V * cos(omega * t),
        v1 + (1-g2) * v2 - (v2 ** 3) / 3 + B2,
        1
    ])

@CM.border(-1)
def border_1 (y, _):
    return y[1] + 1.0

@CM.border(1)
def border_2(y, _):
    return y[1] + 0.1

@CM.border()
def poin_sec (y, param):
    return y[2] - 2 * np.pi / param['omega']

@DM.function(3, 3, ['omega'])
def poin_sec_reset (y, param):
    return y - np.array([0, 0, 2 * np.pi / param['omega']])

@DM.function(2, 3)
def p (y, _):
    return np.append(y, 0)

@DM.function(3, 2)
def p_inv (y, _):
    return y[0:2]

def main ():
    y0 = np.array([-7.739219543281, -0.188608176591])
    param = {
        'k': 0.1,
        'g1': 0.2,
        'g2': 2.0,
        'B1': 8,
        'B2': 5,
        'V0': 0,
        'V': 1,
        'omega': 1.26
    }

    all_modes = (
        CM('alp_1', alp_1, [border_1, poin_sec]),
        CM('alp_2', alp_2, [border_2, poin_sec]),
        DM('pr1', poin_sec_reset),
        DM('pr2', poin_sec_reset),
        DM('p1', p), DM('pinv1', p_inv),
        DM('p2', p), DM('pinv2', p_inv)
    )

    trans = {
        'p1': 'alp_1',
        'p2': 'alp_2',
        'alp_1': ['alp_2', 'pr1'],
        'alp_2': ['alp_1', 'pr2'],
        'pr1': 'pinv1',
        'pr2': 'pinv2',
        'pinv1': 'p1',
        'pinv2': 'p2'
    }

    pmap = PoincareMap(all_modes, trans, 'p1', calc_jac=True, calc_hes=True)
    print(pmap.image_detail(y0, param, 2))

if __name__=="__main__":
    main()