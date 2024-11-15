import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import matplotlib.pyplot as plt

import BSpline

def sigmoid(x_array, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x_array  - x_center)/x_scale)) + y_center

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

def patchOneTimestep(u0, v0, n_teeth, n_gaps, dx_gap, dx_tooth, dx, dt, T_patch, params):
    #n_teeth = len(u0)
    #n_gaps = n_teeth - 1
    #dx_gap = (1.0 - n_teeth * dx_tooth) / (n_teeth - 1)
    #n_points_per_tooth = len(u0[0])
    #n_points_per_gap = int(dx_gap / dx_tooth) * n_points_per_tooth

    # Build the interpolating spline
    teeth_mid_points = 0.5 * dx_tooth + np.linspace(0.0, n_gaps, n_teeth) * (dx_gap + dx_tooth)
    teeth_u_avg_values = np.array([np.average(u0[i]) for i in range(n_teeth)])
    teeth_v_avg_values = np.array([np.average(v0[i]) for i in range(n_teeth)])
    u_spline = BSpline.ClampedCubicSpline(teeth_mid_points, teeth_u_avg_values)
    v_spline = BSpline.ClampedCubicSpline(teeth_mid_points, teeth_v_avg_values)

    # For each patch: calculate Neumann boundary conditions and simulate
    return_u = []
    return_v = []
    for patch in range(n_teeth):
        left_x = teeth_mid_points[patch] - 0.5 * dx_tooth
        right_x = teeth_mid_points[patch] + 0.5 * dx_tooth
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
    gap_over_tooth_size_ratio = 9
    n_points_per_tooth = 11
    n_points_per_gap = gap_over_tooth_size_ratio * (n_points_per_tooth - 1) - 1
    N = n_teeth * n_points_per_tooth + n_gaps * n_points_per_gap
    dx = L / (N - 1)
    dx_tooth = L / (n_teeth + gap_over_tooth_size_ratio * n_gaps)
    dx_gap = dx_tooth * gap_over_tooth_size_ratio
    print('N =', N, dx_tooth, dx_gap, n_points_per_tooth, n_points_per_gap)

    # Model parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Initial condition - divide over all teeth
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    u_sol = []
    v_sol = []
    for i in range(n_teeth):
        u_sol.append(u0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
        v_sol.append(v0[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth])
    u0_sol = u_sol.copy()
    v0_sol = v_sol.copy()

    # Gap-Tooth Timestepping 
    T = 100.0
    dt = 1.e-5
    T_patch = 10 * dt
    n_patch_steps = int(T / T_patch)
    for k in range(n_patch_steps):
        print('t =', k*T_patch)
        u_sol, v_sol = patchOneTimestep(u_sol, v_sol, n_teeth, n_gaps, dx_gap, dx_tooth, dx, dt, T_patch, params)

    # Plot the solution at final time
    x_plot_array = np.array([])
    for i in range(n_teeth):
        x_plot_array = np.concatenate((x_plot_array, x_array[i * (n_points_per_gap + n_points_per_tooth) : i * (n_points_per_gap + n_points_per_tooth) + n_points_per_tooth]))
    plt.plot(x_plot_array, np.concatenate(u_sol), label='u(x, t=450)')
    plt.plot(x_plot_array, np.concatenate(v_sol), label='v(x, t=450)')
    plt.plot(x_plot_array, np.concatenate(u0_sol), label='u(x, t=0)')
    plt.plot(x_plot_array, np.concatenate(v0_sol), label='v(x, t=0)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    patchTimestepper()