import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import scipy.interpolate as inter
import matplotlib.pyplot as plt

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

def patchTimestepper(u0, v0, dx_tooth, dx, dt, T_patch):
    n_teeth = len(u0)
    n_gaps = n_teeth - 1
    dx_gap = (1.0 - n_teeth * dx_tooth) / (n_teeth - 1)
    n_points_per_tooth = len(u0[0])
    n_points_per_gap = int(dx_gap / dx_tooth) * n_points_per_tooth

    # Build the interpolating spline
    teeth_mid_points = 0.5 * dx_tooth + np.linspace(0.0, n_gaps, n_teeth) * (dx_gap + dx_tooth)
    teeth_u_avg_values = np.array([np.average(u0[i]) for i in range(n_teeth)])
    teeth_v_avg_values = np.array([np.average(v0[i]) for i in range(n_teeth)])
    spline = inter.CubicSpline(teeth_mid_points, )





