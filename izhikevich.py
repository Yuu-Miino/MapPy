import numpy as np
from mappy import ContinuousMode as CM, DiscreteMode as DM, PoincareMap, Mode
from mappy.root import *

Mode.parameters = 5

## Izhikevich neuron model
@CM.function(dimension=2)
def izhikevich (y, param):
    v, u = y
    a, b, _, _, I = param
    return np.array([
        0.04 * (v**2) + 5.0 * v + 140.0 - u + I,
        a * (b * v - u)
    ])

## Firing border
@CM.border(direction = 1)
def fire_border(y, param):
    return y[0] - 30.0

## Firing jump
@DM.function(domain_dimension = 2, codomain_dimension = 2)
def jump(y, param):
    C = np.array([-30 + param[2], param[3]])
    return y + C

## Dimension conversion (p: 1 -> 2, pinv: 2 -> 1)
@DM.function(domain_dimension= 1, codomain_dimension= 2)
def p(y, param):
    return np.array([0, 1]) * y + np.array([param[2], 0])

@DM.function(domain_dimension= 2, codomain_dimension= 1)
def pinv(y, param):
    return np.array([0, 1]) @ y

## Main
def main ():
    y0 = -1.71591635
    param = {
        'a': 0.2,
        'b': 0.2,
        'c': -50.0,
        'd': 2.0,
        'I': 10.0
    }
    param_list = list(param.values())
    param_list = np.array(param_list)

    all_modes = (
        DM('m0', p),
        CM('m1', izhikevich, borders=[fire_border]),
        DM('m2', jump),
        DM('m3', pinv)
    )

    transitions = {
        'm0': 'm1',
        'm1': ['m2'],
        'm2': 'm3',
        'm3': 'm0'
    }

    pmap = PoincareMap(all_modes, transitions, 'm0', calc_jac=True, calc_hes=True)

    res = trace_cycle(pmap, y0, param_list, 4, 4.5, show_progress=True)

    y1 = res[-1]['y']
    p1 = res[-1]['params']
    if not is_type_of(y1, type(y0)) or not is_type_of(p1, type(param_list)):
        raise Exception

    ret = trace_local_bf (
        poincare_map=pmap,
        y0=y1,
        params=p1,
        bf_param_idx=4,
        theta=3.14,
        cnt_param_idx=0,
        end_val= 0.3,
        show_progress=True
    )
    print(len(ret))

if __name__=="__main__":
    main()

