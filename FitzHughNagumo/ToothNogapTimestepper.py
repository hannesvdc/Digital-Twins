import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import matplotlib.pyplot as plt
import argparse

import BSpline
from EulerTimestepper import fhn_euler_timestepper

def sigmoid(x_array, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x_array  - x_center)/x_scale)) + y_center

def fixInitialBCs(u0, v0):
    u0[0] = u0[1]
    u0[-1] = u0[-2]
    v0[0] = v0[1]
    v0[-1] = v0[-2]
    return u0, v0

def fhn_rhs(u, v, dx, params):
    u_left = np.roll(u, -1)
    u_right = np.roll(u, 1)
    v_left = np.roll(v, -1)
    v_right = np.roll(v, 1)

    u_xx = (u_left - 2.0*u + u_right) / dx**2
    v_xx = (v_left - 2.0*v + v_right) / dx**2

    u_rhs = u_xx + u - u**3 - v
    v_rhs = params['delta'] * v_xx + params['eps'] * (u - params['a1']*v - params['a0'])

    return u_rhs, v_rhs

def fhn_euler_neumann_patch(u, v, dx, dt, a, b, params):
    u_rhs, v_rhs = fhn_rhs(u, v, dx, params)
    u_new = u + dt * u_rhs
    v_new = v + dt * v_rhs

    # Apply Neumann boundary conditions
    u_new[0]  = u_new[1]  - a[0] * dx
    u_new[-1] = u_new[-2] + b[0] * dx
    v_new[0]  = v_new[1]  - a[1] * dx
    v_new[-1] = v_new[-2] + b[1] * dx

    return u_new, v_new


def eulerNeumannPathTimestepper(u, v, dx, dt, T, a, b, params):
    N_steps = int(T / dt)
    for _ in range(N_steps):
        u, v = fhn_euler_neumann_patch(u, v, dx, dt, a, b, params)
    return u, v

def patchOneTimestep(u0, v0, x_array, L, n_teeth, dx, dt, T_patch, params):
   
    # Build the interpolating spline based on left- and right endpoints
    middle_index = len(u0[0]) // 2
    x_spline_values = []
    u_spline_values = []
    v_spline_values = []
    for patch in range(n_teeth):
        x_spline_values.append(x_array[patch][middle_index])
        u_spline_values.append(u0[patch][middle_index])
        v_spline_values.append(v0[patch][middle_index])
    u_spline = BSpline.ClampedCubicSpline(x_spline_values, u_spline_values, left_bc=0.0, right_bc=L, solver='krylov')
    v_spline = BSpline.ClampedCubicSpline(x_spline_values, v_spline_values, left_bc=0.0, right_bc=L, solver='krylov')

    # For each tooth: calculate Neumann boundary conditions and simulate in that tooth
    return_u = []
    return_v = []
    for patch in range(n_teeth):
        left_x = max(0.0, x_array[patch][0])
        right_x = min(L, x_array[patch][-1])
        a = [u_spline.derivative(left_x), v_spline.derivative(left_x)]
        b = [u_spline.derivative(right_x), v_spline.derivative(right_x)]

        u_new, v_new = eulerNeumannPathTimestepper(u0[patch], v0[patch], dx, dt, T_patch, a, b, params)
        return_u.append(u_new)
        return_v.append(v_new)

    return return_u, return_v

def psiPatchNogap(z0, x_array, L, n_teeth, dx, dt, T_patch, T, params):
    len_uv = len(z0) // 2
    n_points_per_tooth = len_uv // n_teeth

    # Convert the numpy array to the patches datastructure
    u0 = z0[0:len_uv]
    v0 = z0[len_uv:]
    u_patches = []
    v_patches = []
    for i in range(n_teeth):
        u_patches.append(u0[i * n_points_per_tooth : (i+1) * n_points_per_tooth])
        v_patches.append(v0[i * n_points_per_tooth : (i+1) * n_points_per_tooth])

    # Do time-evolution over an interval of size T.
    n_steps = int(T / T_patch)
    for _ in range(n_steps):
        u_patches, v_patches = patchOneTimestep(u_patches, v_patches, x_array, L, n_teeth, dx, dt, T_patch, params)

    # Convert patches datastructure back to a single numpy array
    u_new = np.concatenate(u_patches)
    v_new = np.concatenate(v_patches)
    z_new = np.concatenate((u_new, v_new))

    # Return the psi - function
    return (z0 - z_new) / T

def patchTimestepper(plot=True):
    # Domain parameters
    L = 20.0
    n_teeth = 21
    n_points_per_tooth = 10
    N = n_teeth * n_points_per_tooth
    dx = L / (N - 1)

    # Model parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Initial condition - divide over all teeth
    x_array = np.linspace(0.0, L, N)
    x_plot_array = []
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    u0, v0 = fixInitialBCs(u0, v0)
    u_sol = []
    v_sol = []
    for i in range(n_teeth):
        u_sol.append(u0[i * n_points_per_tooth : (i+1) *  n_points_per_tooth])
        v_sol.append(v0[i * n_points_per_tooth : (i+1) *  n_points_per_tooth])
        x_plot_array.append(x_array[i * n_points_per_tooth : (i+1) * n_points_per_tooth])

    # Gap-Tooth Timestepping 
    T = 450.0
    dt = 1.e-3
    T_patch = 10 * dt
    n_patch_steps = int(T / T_patch)
    for k in range(n_patch_steps):
        if k % 1000 == 0:
            print('t =', round(k*T_patch, 4))
        u_sol, v_sol = patchOneTimestep(u_sol, v_sol, x_plot_array, L, n_teeth, dx, dt, T_patch, params)

    # Calculate psi - value
    T_psi = 1.0
    z_sol = np.concatenate((np.concatenate(u_sol), np.concatenate(v_sol)))
    psi_val = psiPatchNogap(z_sol, x_plot_array, L, n_teeth, dx, dt, T_patch, T_psi, params)
    print('psi', lg.norm(psi_val), np.max(np.abs(psi_val)))

    if plot is False:
        return z_sol

    # Euler Timestepping for Comparison
    u_euler, v_euler = fhn_euler_timestepper(u0, v0, dx, dt, T, params, verbose=False)

    # Plot the solution at final time
    for i in range(n_teeth):
        if i == 0:
            plt.plot(x_plot_array[i], u_sol[i], label=r'$u(x, t=$' + str(T) + r'$)$', color='blue')
            plt.plot(x_plot_array[i], v_sol[i], label=r'$v(x, t=$' + str(T) + r'$)$', color='orange')
        else:
            plt.plot(x_plot_array[i], u_sol[i], color='blue')
            plt.plot(x_plot_array[i], v_sol[i], color='orange')
    plt.plot(x_array, u_euler, label=r'Reference $u(x, t=$' + str(T) + r'$)$', linestyle='dashed', color='green')
    plt.plot(x_array, v_euler, label=r'Reference $v(x, t=$' + str(T) + r'$)$', linestyle='dashed', color='red')
    plt.legend()
    plt.show()

def findSteadyStateNewtonGMRES(return_ss=False):
    # Domain parameters
    L = 20.0
    n_teeth = 21
    n_points_per_tooth = 10
    N = n_teeth * n_points_per_tooth
    dx = L / (N - 1)

    # Model parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Initial condition - divide over all teeth
    x_array = np.linspace(0.0, L, N)
    x_plot_array = []
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    u0, v0 = fixInitialBCs(u0, v0)
    for i in range(n_teeth):
        x_plot_array.append(x_array[i * n_points_per_tooth : (i+1) * n_points_per_tooth])
    
    # Gap-Tooth Psi Function 
    T_psi = 1.0
    dt = 1.e-3
    T_patch = 10 * dt
    psi = lambda z: psiPatchNogap(z, x_plot_array, L, n_teeth, dx, dt, T_patch, T_psi, params)

    # Do Euler timestepping and calculate psi(euler steady - state) for a good initial condition
    u_euler, v_euler = fhn_euler_timestepper(u0, v0, dx, dt, 100.0, params, verbose=False)
    z_euler = np.concatenate((u_euler, v_euler))
    print('Psi Euler', lg.norm(psi(z_euler)))

    # Do Newton - GMRES to find psi(z) = 0 
    tolerance = 1.e-14
    cb = lambda x, f: print(lg.norm(f))
    try:
        z_ss = opt.newton_krylov(psi, z_euler, f_tol=tolerance, method='gmres', callback=cb, maxiter=200)
    except opt.NoConvergence as err:
        str_err = str(err)
        str_err = str_err[1:len(str_err)-1]
        z_ss = np.fromstring(str_err, dtype=float, sep=' ')

    if return_ss:
        return np.concatenate(x_plot_array), z_ss

    z_timestepper = patchTimestepper(plot=False)

    # Plot the steady-state
    u_ss = z_ss[0:N]
    v_ss = z_ss[N:]
    u_ts = z_timestepper[0:N]
    v_ts = z_timestepper[N:]
    plt.plot(x_array, u_ts, linestyle='dashed', label=r'Time Evolution $u(x)$')
    plt.plot(x_array, v_ts, linestyle='dashed', label=r'Time Evolution $v(x)$')
    plt.plot(x_array, u_ss, linestyle='dotted', label=r'Newton - GMRES $u(x)$')
    plt.plot(x_array, v_ss, linestyle='dotted', label=r'Newton - GMRES $v(x)$')
    plt.title('Steady - State Patches without Gaps')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u, v$', rotation=0)
    plt.legend()
    plt.show()

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', dest='experiment', nargs='?')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    
    if args.experiment == 'ss':
        findSteadyStateNewtonGMRES()
    elif args.experiment == 'evolution':
        patchTimestepper()