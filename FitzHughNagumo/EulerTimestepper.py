import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
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

def fhn_euler(u, v, dx, dt, params):
    u_rhs, v_rhs = fhn_rhs(u, v, dx, params)
    u_new = u + dt * u_rhs
    v_new = v + dt * v_rhs

    # Apply homogeneous Neumann boundary conditions
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    v_new[0] = v_new[1]
    v_new[-1] = v_new[-2]

    return u_new, v_new

def fhn_euler_timestepper(u, v, dx, dt, T, params):
    for _ in range(int(T / dt)):
        u, v = fhn_euler(u, v, dx, dt, params)
    return u, v

def psi(x, T, dx, dt, params):
    N = x.size // 2
    u, v = x[0:N], x[N:]

    u_new, v_new = fhn_euler_timestepper(u, v, dx, dt, T, params)
    return np.concatenate((u - u_new, v - v_new)) / T

def plotFitzHughNagumoSolution():
    # Model parameters
    L = 20.0
    N = 200
    dx = L / N
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1 # 0.01 originally for the spatio-temporal oscillations
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Initial condition
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    u = np.copy(u0)
    v = np.copy(v0)

    # Timestepping
    dt = 0.001
    report_dt = 10*dt
    T = 450.0
    u_solution = np.zeros((int(T / report_dt)+1, u.size))
    v_solution = np.zeros((int(T / report_dt)+1, u.size))
    u_solution[0,:] = u0
    v_solution[0,:] = v0
    for n in range(int(T / dt)):
        u, v = fhn_euler(u, v, dx, dt, params)

        if n > 0 and n % 10 == 0:
            u_solution[n // 10, :] = u
            v_solution[n // 10, :] = v

    # Plotting the final result
    x_plot_array = np.linspace(0.0, T, u_solution.shape[1]+1)
    t_plot_array = np.linspace(0.0, T, u_solution.shape[0]+1)
    plt.plot(x_array, u, label='u(x, t=450)')
    plt.plot(x_array, v, label='v(x, t=450)')
    plt.plot(x_array, u0, label='u(x, t=0)')
    plt.plot(x_array, v0, label='v(x, t=0)')
    plt.legend()

    X, Y = np.meshgrid(x_plot_array, t_plot_array)
    v_solution = sigmoid(v_solution, y_scale=2.0, y_center=-1.0, x_scale=0.05)
    u_max = np.max(u_solution)
    u_min = np.min(u_solution)
    v_max = np.max(v_solution)
    v_min = np.min(v_solution)
    print(u_max, u_min, v_max, v_min)
    print('psi', lg.norm(psi(np.concatenate((u,v)), 0.1, delta, a0, a1, eps, dx, dt)))
    plt.figure()
    plt.pcolor(X, Y, u_solution, cmap='viridis', vmin=min(u_min, v_min), vmax=max(u_max, v_max))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(r'$u(x, t)$')
    plt.figure()
    plt.pcolor(X, Y, v_solution, cmap='viridis', vmin=min(u_min, v_min), vmax=max(u_max, v_max))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$t$')
    plt.title(r'$v(x, t)$')
    plt.show()

def findSteadyState():
    # Model parameters
    L = 20.0
    N = 200
    dx = L / N
    dt = 0.001
    T = 0.1
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1 # eps = 0.01 is unstable (oscillatory) beyond the Hopf bifurcation. Doing stable for now.
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    def cb(x, f): 
        print(lg.norm(f))

    F = lambda x: psi(x, T, dx, dt, params)
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    x0 = np.concatenate((u0, v0))
    try:
        x_ss = opt.newton_krylov(F, x0, callback=cb, rdiff=1.e-8, verbose=True, f_tol=1.e-6)
    except opt.NoConvergence as err:
        str_err = str(err)
        str_err = str_err[1:len(str_err)-1]
        x_ss = np.fromstring(str_err, dtype=float, sep=' ')

    u_ss = x_ss[0:N]
    v_ss = x_ss[N:]
    x_array = np.linspace(0.0, L, N)
    plt.plot(x_array, u_ss, label=r'$u(x, t=\infty)$')
    plt.plot(x_array, v_ss, label=r'$v(x, t=\infty)$')
    plt.plot(x_array, u0, label=r'$u(x, t=0)$')
    plt.plot(x_array, v0, label=r'$v(x, t=0)$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plotFitzHughNagumoSolution()
    #findSteadyState()