import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import argparse

from ToothNogapTimestepper import psiPatchNogap, sigmoid

# Model Parameters
a0 = -0.03
a1 = 2.0
delta = 4.0
params = {'delta': 4.0, 'a0': -0.03, 'a1': 2.0}

# Spatial Discretization Parameters
L = 20.0
n_teeth = 21
n_points_per_tooth = 10
N = n_teeth * n_points_per_tooth
dx = L / (N - 1)
x_array = np.linspace(0.0, L, N)

# Time Discretization Parameters
T_psi = 1.0
dt = 1.e-3
T_patch = 10 * dt

# z = (u. v) on a fixed grid
def G(z, eps):
    params['eps'] = eps
    return psiPatchNogap(z, x_array, L, n_teeth, dx, dt, T_patch, T_psi, params) 

def dGdz_w(z, w, eps):
    rdiff = 1.e-8
    return (G(z + rdiff * w, eps) - G(z, eps)) / rdiff

def dGdeps(z, eps):
    rdiff = 1.e-8
    return (G(z, eps + rdiff) - G(z, eps)) / rdiff

# Calculates the tangent to the path at the current point as (Gx^{-1} G_eps, -1).
def computeTangent(Gx_v, G_eps, prev_tangent, M, tolerance):
    x0 = prev_tangent[0:M] / prev_tangent[M]
    A = slg.LinearOperator(matvec=Gx_v, shape=(M,M))

    _tangent = slg.gmres(A, G_eps, x0=x0, atol=tolerance)[0]
    tangent = np.append(_tangent, -1.0)
    tangent = tangent / lg.norm(tangent)

    if np.dot(tangent, prev_tangent) > 0:
        return tangent
    else:
        return -tangent