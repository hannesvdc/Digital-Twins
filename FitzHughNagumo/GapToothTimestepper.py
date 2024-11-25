import numpy as np
import numpy.linalg as lg
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import argparse

import BSpline
from EulerTimestepper import fhn_euler_timestepper

directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'

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

def fhn_euler_patch(u, v, dx, dt, a, b, params):
    u_rhs, v_rhs = fhn_rhs(u, v, dx, params)
    u_new = u + dt * u_rhs
    v_new = v + dt * v_rhs

    # Apply homogeneous Neumann boundary conditions
    u_new[0]  = u_new[1]  - a[0] * dx
    u_new[-1] = u_new[-2] + b[0] * dx
    v_new[0]  = v_new[1]  - a[1] * dx
    v_new[-1] = v_new[-2] + b[1] * dx

    return u_new, v_new

def eulerNeumannPathTimestepper(u, v, dx, dt, T, a, b, params):
    N_steps = int(T / dt)
    for _ in range(N_steps):
        u, v = fhn_euler_patch(u, v, dx, dt, a, b, params)
    return u, v

def patchOneTimestep(u0, v0, x_array, L, n_teeth, dx, dt, T_patch, params, solver='lu_direct'):
   
    # Build the interpolating spline based on left- and right endpoints
    x_spline_values = []
    u_spline_values = []
    v_spline_values = []
    for patch in range(n_teeth):
        x_spline_values.extend([x_array[patch][0], x_array[patch][-1]])
        u_spline_values.extend([u0[patch][0], u0[patch][-1]])
        v_spline_values.extend([v0[patch][0], v0[patch][-1]])
    u_spline = BSpline.ClampedCubicSpline(x_spline_values, u_spline_values, left_bc=0.0, right_bc=L, solver=solver)
    v_spline = BSpline.ClampedCubicSpline(x_spline_values, v_spline_values, left_bc=0.0, right_bc=L, solver=solver)

    # x_plot_array = np.linspace(0.0, L, 1001)
    # plt.plot(x_plot_array, u_spline(x_plot_array), color='tab:green', linestyle='dashed', label='Spline Interpolation')
    # plt.plot(x_plot_array, v_spline(x_plot_array), color='tab:red', linestyle='dashed')
    # for patch in range(len(u0)):
    #     if patch == 0:
    #         plt.plot(x_array[patch], u0[patch], color='tab:blue', label=r'$u(x)$ Patches')
    #         plt.plot(x_array[patch], v0[patch], color='tab:orange', label=r'$u(x)$ Patches')
    #     else:
    #         plt.plot(x_array[patch], u0[patch]+0.01, color='tab:blue')
    #         plt.plot(x_array[patch], v0[patch]+0.01, color='tab:orange')
    # plt.xlabel(r'$x$')
    # plt.legend()
    # plt.show()

    # For each tooth: calculate Neumann boundary conditions and simulate in that tooth
    return_u = []
    return_v = []
    for patch in range(n_teeth):
        left_x = x_array[patch][0]
        right_x = x_array[patch][-1]
        a = [u_spline.derivative(left_x), v_spline.derivative(left_x)]
        b = [u_spline.derivative(right_x), v_spline.derivative(right_x)]

        u_new, v_new = eulerNeumannPathTimestepper(u0[patch], v0[patch], dx, dt, T_patch, a, b, params)
        return_u.append(u_new)
        return_v.append(v_new)

    return return_u, return_v

def patchTimestepper():
    BSpline.ClampedCubicSpline.lu_exists = False

    # Domain parameters
    L = 20.0
    n_teeth = 30#100
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
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
        u_sol.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        v_sol.append(v0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])

    # Gap-Tooth Timestepping 
    T = 200.0
    dt = 1.e-4 # 1.e-5 for n_teeth=100
    T_patch = 10 * dt
    n_patch_steps = int(T / T_patch)
    for k in range(n_patch_steps):
        if k % 1000 == 0:
            print('t =', round(k*T_patch, 4))
        u_sol, v_sol = patchOneTimestep(u_sol, v_sol, x_plot_array, L, n_teeth, dx, dt, T_patch, params, solver='lu_direct')

    # Store the steady - state
    np.save(directory + 'gaptooth_evolution_nteeth='+str(n_teeth) + '_T=' + str(T) + '.npy', np.vstack((np.concatenate(x_plot_array), np.concatenate(u_sol), np.concatenate(v_sol))))

    # Calculate the psi - value of the GapTooth scheme
    T_psi = 1.0
    z_sol = np.concatenate((np.concatenate(u_sol), np.concatenate(v_sol)))
    psi_val = psiPatch(z_sol, x_plot_array, L, n_teeth, dx, dt, T_patch, T_psi, params)
    print('Psi Gap-Tooth', lg.norm(psi_val))

    # Euler Timestepping for Comparison (Keep this - we need to compare at the same end time T)
    print('Running the Euler method for comparison')
    dt_euler = 1.e-3
    N_euler = 200
    dx_euler = L / N_euler
    x_array_euler = np.linspace(0.0, L, N_euler)
    u0_euler = sigmoid(x_array_euler, 6.0, -1, 1.0, 2.0)
    v0_euler = sigmoid(x_array_euler, 10, 0.0, 2.0, 0.1)
    u0_euler, v0_euler = fixInitialBCs(u0_euler, v0_euler)
    u_euler, v_euler = fhn_euler_timestepper(u0_euler, v0_euler, dx_euler, dt_euler, T, params, verbose=False)

    # Plot the solution at final time
    plt.plot(x_plot_array[i], u_sol[i], label=r'$u(x, t=$' + str(T) + r'$)$', color='blue')
    plt.plot(x_plot_array[i], v_sol[i], label=r'$v(x, t=$' + str(T) + r'$)$', color='orange')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_sol[i], color='blue')
        plt.plot(x_plot_array[i], v_sol[i], color='orange')
    plt.plot(x_array_euler, u_euler, label=r'Reference $u(x, t=$' + str(T) + r'$)$', linestyle='dashed', color='green')
    plt.plot(x_array_euler, v_euler, label=r'Reference $v(x, t=$' + str(T) + r'$)$', linestyle='dashed', color='red')
    plt.legend()
    plt.show()

def psiPatch(z0, x_array, L, n_teeth, dx, dt, T_patch, T, params, solver='lu_direct', verbose=False):
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
    for k in range(n_steps):
        if verbose:
            print(k*T_patch)
        u_patches, v_patches = patchOneTimestep(u_patches, v_patches, x_array, L, n_teeth, dx, dt, T_patch, params, solver=solver)

    # Convert patches datastructure back to a single numpy array
    u_new = np.concatenate(u_patches)
    v_new = np.concatenate(v_patches)
    z_new = np.concatenate((u_new, v_new))

    # Return the psi - function
    return (z0 - z_new) / T

def findSteadyStateNewtonGMRES():
    BSpline.ClampedCubicSpline.lu_exists = False

    # Setup the Domain and its Parameters
    L = 20.0
    n_teeth = 30
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
    dx = L / (N - 1)
    x_array = np.linspace(0.0, L, N)
    x_plot_array = []
    for i in range(n_teeth):
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])

    # Model parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Load the Gap-Tooth Steady State from Time Evolution and use it as initial condition
    print('Loading Initial Condition from File ...')
    gt_data = np.load(directory + 'gaptooth_evolution_nteeth='+str(n_teeth) + '_T=' + str(200.0) + '.npy')
    u_gt = gt_data[1,:]
    v_gt = gt_data[2,:]
    z_gt = np.concatenate((u_gt, v_gt))

    # Calculate the psi - value of the Euler scheme. First transform Euler to the patches datastructure
    print('\nCalculating Initial Psi Value ...')
    dt = 1.e-4
    T_psi = 0.2
    T_patch = 10 * dt
    psi = lambda z: psiPatch(z, x_plot_array, L, n_teeth, dx, dt, T_patch, T_psi, params)
    print('Initial Gap-Tooth Psi', lg.norm(psi(z_gt)))

    # Do Newton - GMRES to find psi(z) = 0
    print('\nCalculating Steady State via Newton-GMRES ...')
    tolerance = 1.e-14
    cb = lambda x, f: print(lg.norm(f))
    try:
        z_ss = opt.newton_krylov(psi, z_gt, f_tol=tolerance, method='gmres', callback=cb, maxiter=200, verbose=True)
    except opt.NoConvergence as err:
        str_err = str(err)
        str_err = str_err[1:len(str_err)-1]
        z_ss = np.fromstring(str_err, dtype=float, sep=' ')
    N_ss = len(z_ss) // 2
    u_ss = z_ss[0:N_ss]
    v_ss = z_ss[N_ss:]

    # Store the steady - state
    print('Storing and Plotting ...')
    np.save(directory + 'gaptooth_steady_state.npy', np.vstack((np.concatenate(x_plot_array), u_ss, v_ss)))

    # Convert the found steady-state to the gap-tooth datastructure and plot
    u_patch_ss = []
    v_patch_ss = []
    u_patch_gt = []
    v_patch_gt = []
    for i in range(n_teeth):
        u_patch_ss.append(u_ss[i * n_points_per_tooth : (i+1) * n_points_per_tooth])
        v_patch_ss.append(v_ss[i * n_points_per_tooth : (i+1) * n_points_per_tooth])
        u_patch_gt.append(u_gt[i * n_points_per_tooth : (i+1) * n_points_per_tooth])
        v_patch_gt.append(v_gt[i * n_points_per_tooth : (i+1) * n_points_per_tooth])

    plt.plot(x_plot_array[i], u_patch_ss[i], linestyle='dashdot', label=r'Newton - GMRES $u(x)$', color='blue')
    plt.plot(x_plot_array[i], v_patch_ss[i], linestyle='dashdot', label=r'Newton - GMRES $v(x)$', color='orange')
    plt.plot(x_plot_array[i], u_patch_gt[i], linestyle='dotted', label=r'Gap-Tooth $u(x)$', color='green')
    plt.plot(x_plot_array[i], v_patch_gt[i], linestyle='dotted', label=r'Gap-Tooth $v(x)$', color='red')
    for i in range(1, n_teeth):
        plt.plot(x_plot_array[i], u_patch_ss[i], color='blue')
        plt.plot(x_plot_array[i], v_patch_ss[i], color='orange')
        plt.plot(x_plot_array[i], u_patch_gt[i], linestyle='dotted', color='green')
        plt.plot(x_plot_array[i], v_patch_gt[i], linestyle='dotted', color='red')
    plt.title('Steady-State Gap-Tooth')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$u, v$', rotation=0)
    plt.legend()
    plt.show()

def calculateEigenvalues():
    BSpline.ClampedCubicSpline.lu_exists = False

    # Setup the Domain and its Parameters
    L = 20.0
    n_teeth = 100
    n_gaps = n_teeth - 1
    gap_over_tooth_size_ratio = 1
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
    dx = L / (N - 1)
    x_array = np.linspace(0.0, L, N)
    x_plot_array = []
    for i in range(n_teeth):
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])

    # Model parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Load the Gap-Tooth Steady State from Time Evolution and use it as initial condition
    print('Loading Steady State from File ...')
    ss_data = np.load(directory + 'gaptooth_evolution_T=' + str(200.0) + '.npy')
    u_ss = ss_data[1,:]
    v_ss = ss_data[2,:]
    z_ss = np.concatenate((u_ss, v_ss))

    # Gap-Tooth Psi Function
    print('\nCalculating Leading Eigenvalues using Arnoldi ...')
    r_diff = 1.e-8
    T_psi = 0.2
    dt = 1.e-3
    T_patch = 10 * dt
    psi = lambda z: psiPatch(z, x_plot_array, L, n_teeth, dx, dt, T_patch, T_psi, params, solver='lu_direct')
    d_psi_mvp = lambda w: (psi(z_ss + r_diff * w) - psi(z_ss)) / r_diff
    D_psi = slg.LinearOperator(shape=(2*N, 2*N), matvec=d_psi_mvp)
    psi_eigvals = slg.eigs(D_psi, k=2*N-2, which='LM', return_eigenvectors=False)
    print('Done.')

    # Save the eigenvalues to file
    #np.save(directory + 'tooth_no_gap_eigenvalues.npy', psi_eigvals)

    # Calculate the eigenvalues of the right-hand side PDE in the grid points as well
    # Use those stored on file as an approximation
    euler_eigvals = np.load(directory + 'euler_eigenvalues.npy')
    euler_eigvals = euler_eigvals[1,:]
    #approx_psi_eigvals = 1.0 - np.exp(f_eigvals * T_psi)

    # Plot the eigenvalues in the complex plane
    plt.scatter(np.real(psi_eigvals), np.imag(psi_eigvals), label='Timestepper Eigenvalues')
    plt.scatter(np.real(euler_eigvals), np.imag(euler_eigvals), label=r'Euler Timestepper Eigenvalues')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('Gap-Tooth Timestepper Eigenvalues')
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
    else:
        print('Select either --experiment ss or --experiment evolution')