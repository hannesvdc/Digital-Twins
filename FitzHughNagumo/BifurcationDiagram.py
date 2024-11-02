import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import matplotlib.pyplot as plt

from EulerTimestepper import psi

N = 200
L = 20.0
dx = L / N
dt = 0.001
T = 1.0
params = {'delta': 4.0, 'a0': -0.03, 'a1': 2.0, 'eps': 0.0}

def G(x, eps):
    params['eps'] = eps
    return psi(x, T, dx, dt, params)

def dGdx_v(x, v, eps):
    rdiff = 1.e-8
    norm_v = lg.norm(v)
    return (G(x + rdiff * v / norm_v, eps) - G(x, eps)) / rdiff

def dGdeps(x, eps):
    rdiff = 1.e-8
    return (G(x, eps + rdiff) - G(x, eps)) / rdiff