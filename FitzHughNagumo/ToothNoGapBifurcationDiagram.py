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
    
"""
The Internal Numerical Continuation Routine.
"""
# Variable z = (u, v)
def numericalContinuation(z0, eps0, initial_tangent, max_steps, ds, ds_min, ds_max, tolerance):
    M = 2*N

    z = np.copy(z0)
    eps = eps0
    prev_tangent = np.copy(initial_tangent)

    z_path = [np.copy(z)]
    eps_path = [eps]
    print_str = 'Step {0:3d}:\t <u>: {1:4f}\t eps: {2:4f}\t ds: {3:6f}'.format(0, np.mean(z_path[0][0:N]), eps, ds)
    print(print_str)

    for n in range(1, max_steps+1):
		# Calculate the tangent to the curve at current point 
        Gz_w = lambda w: dGdz_w(x, w, eps)
        Geps = dGdeps(z, eps)
        tangent = computeTangent(Gz_w, Geps, prev_tangent, M, tolerance)

		# Create the extended system for corrector: q = (z, eps) = (u, v, eps)
        N_opt = lambda q: np.dot(tangent, q - np.append(z, eps)) + ds
        F = lambda q: np.append(G(q[0:M], q[M]), N_opt(q))

		# Our implementation uses adaptive timetepping
        while ds > ds_min:
			# Predictor: Extrapolation
            z_p = z + ds * tangent[0:M]
            eps_p = eps + ds * tangent[M]
            q_p = np.append(z_p, eps_p)

			# Corrector: Newton - Krylov
            try:
                q_new = opt.newton_krylov(F, q_p, f_tol=tolerance)
                z = q_new[0:M]
                eps = q_new[M]
                z_path.append(np.copy(z))
                eps_path.append(eps)

				# Updating the arclength step and tangent vector
                ds = min(1.2*ds, ds_max)
                prev_tangent = np.copy(tangent)

                break
            except:
                # Decrease arclength if the corrector fails.
                ds = max(0.5*ds, ds_min)
        else:
            print('Minimal Arclength Size is too large. Aborting.')
            return z_path, eps_path
		
        print_str = 'Step {0:3d}:\t <u>: {1:4f}\t eps: {2:4f}\t ds: {3:6f}'.format(n, np.mean(z_path[-1][0:N]), eps, ds)
        print(print_str)

    return z_path, eps_path