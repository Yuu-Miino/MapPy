import numpy as np
from hds import ContinuousMode as CM, DiscreteMode as DM, solve_ivbmp
from hds.tools import DictVector

class Parameter(DictVector):
    a: float = 0.2
    b: float = 0.2
    c: float = -50
    d: float = 2.0
    I: float = 10

## Izhikevich neuron model
@CM.function(dimension=2)
def izhikevich (y, param):
    v, u = y
    a = param.a
    b = param.b
    I = param.I

    return np.array([
        0.04 * (v**2) + 5.0 * v + 140.0 - u + I,
        a * (b * v - u)
    ])

def izhikevich_jac (y, param: Parameter):
    v = y[0]
    a = param.a
    b = param.b
    return np.array([
        [0.08 * v + 5, -1],
        [a * b, -a]
    ])

def izhikevich_hes (y, param: Parameter):
    return np.array([
        [
            [0.08, 0],
            [0, 0]
        ],
        [
            [0, 0],
            [0, 0]
        ]
    ])

## Firing border
@CM.border(direction = 1)
def fire_border(y, param):
    return y[0] - 30.0

def fire_border_dy(y, param):
    return np.array([1, 0])

def fire_border_dy2(y, param):
    return np.array([[0, 0], [0, 0]])

## Firing jump
@DM.function(domain_dimension = 2, codomain_dimenstion = 2)
def jump(y, param):
    C = np.array([-30 + param.c, param.d])
    return y + C

def jump_jac (y, param):
    return np.eye(2)

def jump_hes (y, param):
    return np.zeros((2,2,2))

## Dimension conversion (p: 1 -> 2, pinv: 2 -> 1)
@DM.function(domain_dimension= 1, codomain_dimenstion= 2)
def p(y, param):
    return np.array([0, 1]) * y + np.array([param.c, 0])

def p_jac(y, param):
    return np.array([0, 1])

def p_hes(y, param):
    return np.array([0, 0])

@DM.function(domain_dimension= 2, codomain_dimenstion= 1)
def pinv(y, param):
    return float(np.array([0, 1]) @ y)

def pinv_jac(y, param) -> np.ndarray:
    return np.array([0, 1])

def pinv_hes(y, param) -> np.ndarray:
    return np.array([[0, 0], [0, 0]])

## Main
def main ():
    y0 = -1.71591635
    param = Parameter()

    mode0 = DM(p, jac_fun=p_jac, hes_fun=p_hes)
    mode1 = CM(izhikevich, borders=[fire_border],
        jac_fun=izhikevich_jac, hes_fun=izhikevich_hes,
        jac_border= [fire_border_dy], hes_border=[fire_border_dy2])
    mode2 = DM(jump, jac_fun=jump_jac, hes_fun=jump_hes)
    mode3 = DM(pinv, jac_fun=pinv_jac, hes_fun=pinv_hes)

    mode0.next = mode1
    mode1.next = [mode2]
    mode2.next = mode3
    mode3.next = mode0

    result = solve_ivbmp(y0, mode0, args=param, calc_hes=True)
    print(result)

if __name__=="__main__":
    main()
