import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from EulerTimestepper import psi, sigmoid

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
    return (G(x + rdiff * v, eps) - G(x, eps)) / rdiff

def dGdeps(x, eps):
    rdiff = 1.e-8
    return (G(x, eps + rdiff) - G(x, eps)) / rdiff

def calculateMatrix(func, M):
    e = np.eye(M)
    matrix = np.zeros((M, M))
    for m in range(M):
        matrix[:,m] = func(e[:,m])
    return matrix

def computeTangent(Gx_v, G_eps, prev_tangent, M, tolerance):
    DG_matrix = calculateMatrix(Gx_v, M)
    tangent = np.append(lg.solve(DG_matrix, G_eps), -1.0) # Solve using gmres
    tangent = tangent / lg.norm(tangent)
    print('tangent', tangent)

    if np.dot(tangent, prev_tangent) > 0:
        return tangent
    else:
        return -tangent


def numericalContinuation(x0, eps0, max_steps, ds, ds_min, ds_max, tolerance):
    M = 2*N
    x = np.copy(x0)
    eps = eps0
    x_path = [np.mean(x[0:N])]
    eps_path = [eps]

    print_str = 'Step n: {0:3d}\t <u>: {1:4f}\t eps: {2:4f}'.format(0, x_path[0], eps)
    print(print_str)

	# Choose intial tangent (guess). No idea why yet, but we need to
	# negate the tangent to find the actual search direction
    rng = rd.RandomState()
    random_tangent = rng.normal(0.0, 1.0, M+1)
    initial_tangent = computeTangent(lambda v: dGdx_v(x0, v, eps0), dGdeps(x0, eps0), random_tangent/lg.norm(random_tangent), M, tolerance)
    prev_tangent = -initial_tangent / lg.norm(initial_tangent)

    for n in range(1, max_steps+1):
		# Determine the tangent to the curve at current point
		# By solving an underdetermined system with quadratic constraint norm(tau)**2 = 1
        Gx_v = lambda v: dGdx_v(x, v, eps)
        Geps = dGdeps(x, eps)
        tangent = computeTangent(Gx_v, Geps, prev_tangent, M, tolerance)

		# Create the extended system for corrector: z = (x, eps) = (u, v, eps)
        N_opt = lambda z: np.dot(tangent, z - np.append(x, eps)) + ds
        F = lambda z: np.append(G(z[0:M], z[M]), N_opt(z))

		# Our implementation uses adaptive timetepping
        while ds > ds_min:
			# Predictor: Extrapolation
            x_p = x + ds * tangent[0:M]
            eps_p = eps + ds * tangent[M]
            z_p = np.append(x_p, eps_p)

			# Corrector: Newton - Krylov
            try:
                z_new = opt.newton_krylov(F, z_p, f_tol=tolerance)
                x = z_new[0:M]
                eps = z_new[M]
                x_path.append(np.mean(x[0:N]))
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
		
        print_str = 'Step n: {0:3d}\t <x>: {1:4f}\t eps: {2:4f}'.format(n, np.mean(x[0:N]), eps)
        print(print_str)

    return np.array(x_path), np.array(eps_path)

def plotBifurcationDiagram():
    # Construct the initial point on the path
    eps0 = 0.1
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 14.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 15, 0.0, 2.0, 0.1)
    x0 = np.concatenate((u0, v0))
    F = lambda x: G(x, eps0)
    tolerance = 1.e-12
    x0 = opt.newton_krylov(F, x0, rdiff=1.e-8, f_tol=tolerance)

    # Continuation Parameters
    max_steps = 1000
    ds_min = 1.e-6
    ds_max = 0.01
    ds = 0.001

    # Do actual numerical continuation
    x_path, eps_path = numericalContinuation(x0, eps0, max_steps, ds, ds_min, ds_max, tolerance)

    plt.plot(eps_path, x_path)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$<u>$')

if __name__ == '__main__':
    plotBifurcationDiagram()