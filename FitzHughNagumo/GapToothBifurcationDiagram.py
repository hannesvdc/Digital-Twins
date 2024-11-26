import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import BSpline
BSpline.ClampedCubicSpline.lu_exists = False

from GapToothTimestepper import psiPatch

directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'

# Model Parameters
a0 = -0.03
a1 = 2.0
delta = 4.0
params = {'delta': 4.0, 'a0': -0.03, 'a1': 2.0}

# Domain parameters
L = 20.0
n_teeth = 30
n_gaps = n_teeth - 1
gap_over_tooth_size_ratio = 1
n_points_per_tooth = 11
n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
N_all = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap # Also counts points in the gaps
dx = L / (N_all - 1)
N = n_teeth * n_points_per_tooth

# Initial condition - divide over all teeth
x_array = np.linspace(0.0, L, N_all)
x_patch_array = []
for i in range(n_teeth):
    x_patch_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])

# Time Discretization Parameters
T_psi = 0.2
dt = 1.e-4
T_patch = 10 * dt

# z = (u. v) on a fixed grid
def G(z, eps):
    params['eps'] = eps
    return psiPatch(z, x_patch_array, L, n_teeth, dx, dt, T_patch, T_psi, params, solver='lu_direct', verbose=False) 

def dGdz_w(z, w, eps):
    rdiff = 1.e-8
    return (G(z + rdiff * w, eps) - G(z, eps)) / rdiff

def dGdeps(z, eps):
    rdiff = 1.e-8
    return (G(z, eps + rdiff) - G(z, eps)) / rdiff

# Calculates the tangent to the path at the current point as (Gx^{-1} G_eps, -1).
def computeTangent(Gx_v, G_eps, prev_tangent, M, tolerance):
    z0 = prev_tangent[0:M] / prev_tangent[M]
    A = slg.LinearOperator(matvec=Gx_v, shape=(M,M))

    _tangent = slg.gmres(A, G_eps, x0=z0, atol=tolerance)[0]
    tangent = np.append(_tangent, -1.0)
    tangent = tangent / lg.norm(tangent)

    if np.dot(tangent, prev_tangent) > 0:
        return tangent
    else:
        return -tangent
    
"""
The Internal Numerical Continuation Routine.
"""
# Variable z = (u, v), a single ndarray
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
        Gz_w = lambda w: dGdz_w(z, w, eps)
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

"""
Routine that calculates the bifurcation diagram of a timestepper for the Fitzhugh-Nagumo PDE. Steady states of 
the pde equal fixex points of the timespper, or zeros of psi(x) = (x - phis_T(x)) / T, with phi_T the timestepper.
"""
def calculateBifurcationDiagram():
    eps0 = 0.1
    params['eps'] = eps0
    M = 2 * N
    tolerance = 1.e-6

    # Load the steady state as initial point on the bf diagram
    print('Loading Initial Point on the Bifurcation Diagram ...')
    gt_ss = np.load(directory + 'gaptooth_steady_state.npy')
    u_ss = gt_ss[1,:]
    v_ss = gt_ss[2,:]
    z0 = np.concatenate((u_ss, v_ss))

    # Continuation Parameters
    max_steps = 200
    ds_min = 1.e-6
    ds_max = 0.1
    ds = 0.001

    # Calculate the tangent to the path at the initial condition x0
    rng = rd.RandomState()
    random_tangent = rng.normal(0.0, 1.0, M+1)
    initial_tangent = computeTangent(lambda v: dGdz_w(z0, v, eps0), dGdeps(z0, eps0), random_tangent / lg.norm(random_tangent), M, tolerance)
    initial_tangent = initial_tangent / lg.norm(initial_tangent)

    # Do actual numerical continuation in both directions
    print('Runnning Pseudo-Arclength Continuation ...')
    if initial_tangent[-1] < 0.0: # Decreasing eps
        print('Increasing eps first')
        sign = 1.0
    else:
        sign = -1.0
    z1_path, eps1_path = numericalContinuation(z0, eps0,  sign * initial_tangent, max_steps, ds, ds_min, ds_max, tolerance)
    z2_path, eps2_path = numericalContinuation(z0, eps0, -sign * initial_tangent, max_steps, ds, ds_min, ds_max, tolerance)

    # Store the full path
    z1_path = np.array(z1_path)
    z2_path = np.array(z2_path)
    eps1_path = np.array(eps1_path)
    eps2_path = np.array(eps2_path)
    np.save(directory + 'gaptooth_bf_diagram.npy', np.hstack((z1_path, eps1_path[:,np.newaxis], z2_path, eps2_path[:,np.newaxis])))

    # Plot both branches
    plot_z1_path = np.average(z1_path[:, 0:N], axis=1)
    plot_z2_path = np.average(z2_path[:, 0:N], axis=1)
    plt.plot(eps1_path, plot_z1_path, color='blue', label='Branch 1')
    plt.plot(eps2_path, plot_z2_path, color='red', label='Branch 2')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$<u>$')
    plt.title('Bifurcation Diagram of the Gap-Tooth Scheme')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    calculateBifurcationDiagram()