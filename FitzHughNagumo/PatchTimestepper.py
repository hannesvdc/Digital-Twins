import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import matplotlib.pyplot as plt

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

def patchOneTimestep(u0, v0, x_array, L, n_teeth, dx, dt, T_patch, params):
   
    # Build the interpolating spline based on left- and right endpoints
    x_spline_values = []
    u_spline_values = []
    v_spline_values = []
    for patch in range(n_teeth):
        x_spline_values.extend([x_array[patch][0], x_array[patch][-1]])
        u_spline_values.extend([u0[patch][0], u0[patch][-1]])
        v_spline_values.extend([v0[patch][0], v0[patch][-1]])
    u_spline = BSpline.ClampedCubicSpline(x_spline_values, u_spline_values, left_bc=0.0, right_bc=L)
    v_spline = BSpline.ClampedCubicSpline(x_spline_values, v_spline_values, left_bc=0.0, right_bc=L)

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
    # Domain parameters
    L = 20.0
    n_teeth = 11
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
    u_time_solution = []
    v_time_solution = []
    t_plot_array = [0.0]
    for i in range(n_teeth):
        u_sol.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        v_sol.append(v0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        x_plot_array.append(x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        u_time_solution.append(np.copy(u_sol[i])[np.newaxis, :])
        v_time_solution.append(np.copy(v_sol[i])[np.newaxis, :])

    # Gap-Tooth Timestepping 
    T = 450.0
    dt = 1.e-4
    T_patch = 10 * dt
    n_patch_steps = int(T / T_patch)
    for k in range(n_patch_steps):
        if k % 1000 == 0:
            print('t =', round(k*T_patch, 4))
        u_sol, v_sol = patchOneTimestep(u_sol, v_sol, x_plot_array, L, n_teeth, dx, dt, T_patch, params)
        for i in range(n_teeth):
            u_time_solution[i] = np.vstack([u_time_solution[i], np.copy(u_sol[i])])
            v_time_solution[i] = np.vstack([v_time_solution[i], np.copy(v_sol[i])])
        t_plot_array.append((k+1) * T_patch)

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

    # Plot the full (x, t) evolution diagram for u and v
    t_plot_array = np.array(t_plot_array)
    u_min = min([np.min(u_time_solution[i]) for i in range(n_teeth)])
    u_max = min([np.max(u_time_solution[i]) for i in range(n_teeth)])
    v_min = min([np.min(v_time_solution[i]) for i in range(n_teeth)])
    v_max = min([np.max(v_time_solution[i]) for i in range(n_teeth)])

    plt.figure()
    for i in range(n_teeth):
        X, T = np.meshgrid(x_plot_array[i], t_plot_array)
        plt.pcolor(X, T, u_time_solution[i], cmap='viridis', vmin=min(u_min, v_min), vmax=max(u_max, v_max))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(r'$u(x, t)$')

    plt.figure()
    t_plot_array = np.array(t_plot_array)
    for i in range(n_teeth):
        X, T = np.meshgrid(x_plot_array[i], t_plot_array)
        plt.pcolor(X, T, v_time_solution[i], cmap='viridis', vmin=min(u_min, v_min), vmax=max(u_max, v_max))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(r'$v(x, t)$')

    plt.show()

if __name__ == '__main__':
    patchTimestepper()