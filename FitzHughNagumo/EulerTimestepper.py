import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

import argparse

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

# right-hand side with boundary conditions incorporated
def fhn_rhs_arnoldi(u, v, N, dx, params):
    u_ext = np.hstack([u[0], u, u[-1]])
    v_ext = np.hstack([v[0], v, v[-1]])

    u_left = np.roll(u_ext, -1)[1:N+1]
    u_right = np.roll(u_ext, 1)[1:N+1]
    v_left = np.roll(v_ext, -1)[1:N+1]
    v_right = np.roll(v_ext, 1)[1:N+1]

    u_xx = (u_left - 2.0*u + u_right) / dx**2
    v_xx = (v_left - 2.0*v + v_right) / dx**2

    u_rhs = u_xx + u - u**3 - v
    v_rhs = params['delta'] * v_xx + params['eps'] * (u - params['a1']*v - params['a0'])

    return u_rhs, v_rhs

def f_wrapper(z, N, dx, params):
    u = z[0:N]
    v = z[N:]
    u_rhs, v_rhs = fhn_rhs_arnoldi(u, v, N, dx, params)
    return np.concatenate((u_rhs, v_rhs))

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

def fhn_euler_timestepper(u, v, dx, dt, T, params, verbose=False):
    N_steps = int(T / dt)
    for k in range(N_steps):
        if verbose:
            print('t =', k * dt)
        u, v = fhn_euler(u, v, dx, dt, params)
    return u, v

def psi(x, T, dx, dt, params):
    N = x.size // 2
    u, v = x[0:N], x[N:]

    u_new, v_new = fhn_euler_timestepper(u, v, dx, dt, T, params)
    return np.concatenate((u - u_new, v - v_new)) / T # Not necessary to divide by T, but this works fine. Either way, T_psi = 1.0

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
    dt = 1.e-3
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
    plt.legend()

    X, Y = np.meshgrid(x_plot_array, t_plot_array)
    v_solution = sigmoid(v_solution, y_scale=2.0, y_center=-1.0, x_scale=0.05)
    u_max = np.max(u_solution)
    u_min = np.min(u_solution)
    v_max = np.max(v_solution)
    v_min = np.min(v_solution)
    print('psi', lg.norm(psi(np.concatenate((u,v)), 0.1, dx, dt, params)))
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

def findSteadyState(return_ss=False):
    # Method parameters
    L = 20.0
    N = 200
    dx = L / N
    dt = 1.e-3
    T = 1.0

    # Model parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1 # eps = 0.01 is unstable (oscillatory) beyond the Hopf bifurcation. Doing stable for now.
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 14.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 15, 0.0, 2.0, 0.1)
    x0 = np.concatenate((u0, v0))
    F = lambda x: psi(x, T, dx, dt, params)
    tolerance = 1.e-14
    try:
        cb = lambda x, f: print(lg.norm(f))
        x_ss = opt.newton_krylov(F, x0, rdiff=1.e-8, callback=cb, f_tol=tolerance, method='gmres')
    except opt.NoConvergence as err:
        str_err = str(err)
        str_err = str_err[1:len(str_err)-1]
        x_ss = np.fromstring(str_err, dtype=float, sep=' ')

    if return_ss:
        return x_array, x_ss
    
    # Also do Euler Timestepping for comparison
    u_euler = np.copy(u0)
    v_euler = np.copy(v0)
    for n in range(int(450.0 / dt)):
        u_euler, v_euler = fhn_euler(u_euler, v_euler, dx, dt, params)

    u_ss = x_ss[0:N]
    v_ss = x_ss[N:]
    x_array = np.linspace(0.0, L, N)
    plt.plot(x_array, u_ss, linestyle='dashed', label=r'Newton-GMRES $u(x)$')
    plt.plot(x_array, v_ss, linestyle='dashed', label=r'Newton-GMRES $v(x)$')
    plt.plot(x_array, u_euler+0.005, linestyle='dashdot', label=r'Euler Method $u(x)$')
    plt.plot(x_array, v_euler+0.005, linestyle='dashdot', label=r'Euler Method $v(x)$')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.show()

def calculateEigenvalues():
    # Method parameters
    L = 20.0
    N = 200
    dx = L / N
    dt = 1.e-3
    T_psi = 1.0

    # Model Parameters
    a0 = -0.03
    a1 = 2.0
    delta = 4.0
    eps = 0.1
    params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

    # Do time-evolution to get the steady-state
    print('Calculating the Steady State ...')
    _, z_ss = findSteadyState(return_ss=True)
    f_int = lambda z: f_wrapper(z, N, dx, params)
    z_f_ss = opt.newton_krylov(f_int, z_ss, f_tol=1.e-13)
    print('Done.')

    # Calculate the eigenvalues of Psi in steady state
    print('\nCalculating Leading Eigenvalues of Psi using Arnoldi ...')
    r_diff = 1.e-8
    d_psi_mvp = lambda w: (psi(z_ss + r_diff * w, T_psi, dx, dt, params) - psi(z_ss, T_psi, dx, dt, params)) / r_diff
    D_psi = slg.LinearOperator(shape=(2*N, 2*N), matvec=d_psi_mvp)
    psi_eigvals = slg.eigs(D_psi, k=2*N-2, which='LM', return_eigenvectors=False)
    print('Done.')

    # Calculate the eigenvalues of the PDE right-hand side in steady state
    print('\nCalculating Leading Eigenvalues of the PDE using Arnoldi ...')
    d_f_mvp = lambda w: (f_wrapper(z_f_ss + r_diff * w, N, dx, params) - f_wrapper(z_f_ss, N, dx, params)) / r_diff
    D_f = slg.LinearOperator(shape=(2*N, 2*N), matvec=d_f_mvp)
    f_eigvals = slg.eigs(D_f, k=2*N-2, which='LM', return_eigenvectors=False)
    psi_approx_eigvals = 1.0 - np.exp(f_eigvals * T_psi)
    print('Done.')

    # Plot the Eigenvalues
    plt.scatter(np.real(psi_eigvals), np.imag(psi_eigvals), label=r'Eigenvalues $\mu$ of $\psi$ ')
    plt.scatter(np.real(psi_approx_eigvals), np.imag(psi_approx_eigvals), label=r'$1 - \exp(\sigma T)$')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('Timestepper Eigenvalues')
    plt.legend()

    plt.figure()
    plt.scatter(np.real(f_eigvals), np.imag(f_eigvals), label=r'Eigenvalues $\sigma$ of $f$ ')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(visible=True, which='major', axis='both')
    plt.title('PDE Eigenvalues')
    plt.legend()
    plt.show()


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', dest='experiment', nargs='?')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()

    if args.experiment == 'evolution':
        plotFitzHughNagumoSolution()
    elif args.experiment == 'ss':
        findSteadyState()
    elif args.experiment == 'arnoldi':
        calculateEigenvalues()