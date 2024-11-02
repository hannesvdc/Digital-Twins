import numpy as np
import numpy.linalg as lg
import numpy.random as rd
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

def computeTangent(Gx_v, G_eps, prev_tangent, tolerance):
    DG = lambda v: np.hstack((Gx_v(v[0:N]), G_eps * v[N]))
    g_tangent = lambda v: np.append(DG(v), np.dot(v, v) - 1.0)
    tangent = opt.newton_krylov(g_tangent, prev_tangent, f_tol=tolerance)
	
    return tangent

def numericalContinuation(x0, eps0, max_steps, ds, ds_min, ds_max, tolerance):
    x = np.copy(x0)
    eps = eps0
    x_path = [x]
    eps_path = [eps]

    print_str = 'Step n: {0:3d}\t <u>: {1:4f}\t eps: {2:4f}'.format(0, np.mean(x[0:N]), eps)
    print(print_str)

	# Choose intial tangent (guess). No idea why yet, but we need to
	# negate the tangent to find the actual search direction
    rng = rd.RandomState()
    random_tangent = rng.normal(0.0, 1.0, N+1)
    initial_tangent = computeTangent(lambda v: dGdx_v(x0, v, eps0), dGdeps(x0, eps0), random_tangent/lg.norm(random_tangent), tolerance)
    prev_tangent = -initial_tangent / lg.norm(initial_tangent)

    for n in range(1, max_steps+1):
		# Determine the tangent to the curve at current point
		# By solving an underdetermined system with quadratic constraint norm(tau)**2 = 1
        Gx_v = lambda v: dGdx_v(x, v, eps)
        Geps = dGdeps(x, eps)
        tangent = computeTangent(Gx_v, Geps, prev_tangent, tolerance)

		# Create the extended system for corrector: z = (x, eps) = (u, v, eps)
        N = lambda z: np.dot(tangent, z - np.append(x, eps)) + ds
        F = lambda z: np.append(G(z[0:N], z[N]), N(z))

		# Our implementation uses adaptive timetepping
        while ds > ds_min:
			# Predictor: Extrapolation
            x_p = x + ds * tangent[0:N]
            eps_p = eps + ds * tangent[N]
            z_p = np.append(x_p, eps_p)

			# Corrector: Newton - Krylov
            try:
                z_new = opt.newton_krylov(F, z_p, f_tol=tolerance)
                x = z_new[0:N]
                eps = z_new[N]
                x_path.append(x)
                eps_path.append(eps)

				# Updating the arclength step and tangent vector
                ds = min(1.2*ds, ds_max)
                prev_tangent = np.copy(tangent)

                break
            except:
                # Decrease arclength if Newton routine needs more than max_it iterations
                ds = max(0.5*ds, ds_min)
        else:
			# This case should never happpen under normal circumstances
            print('Minimal Arclength Size is too large. Aborting.')
            return x_path, eps_path
		
        print_str = 'Step n: {0:3d}\t <x>: {1:4f}\t eps: {2:4f}'.format(n, np.mean(x), eps)
        print(print_str)

    return np.array(x_path), np.array(eps_path)