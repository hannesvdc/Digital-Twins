import autograd.numpy as np
import autograd.numpy.linalg as lg
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.sparse.linalg as slg

from autograd import jacobian

# Model parameters
L = 20.0
N = 200
dx = L / N
dt = 1.e-3
a0 = -0.03
a1 = 2.0
delta = 4.0
eps = 0.1 
params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

def sigmoid(x_array, x_center=0.0, y_center=0.0, x_scale=1.0, y_scale=1.0):
    return y_scale / (1.0 + np.exp(-(x_array  - x_center)/x_scale)) + y_center

# w = (u, v)
def fhn_rhs_autograd(w):
    M = len(w) // 2
    u = w[:M]
    v = w[M:]

    u_left = np.roll(u, -1)
    u_right = np.roll(u, 1)
    v_left = np.roll(v, -1)
    v_right = np.roll(v, 1)

    u_xx = (u_left - 2.0*u + u_right) / dx**2
    v_xx = (v_left - 2.0*v + v_right) / dx**2

    u_rhs = u_xx + u - u**3 - v
    v_rhs = params['delta'] * v_xx + params['eps'] * (u - params['a1']*v - params['a0'])

    return np.concatenate((u_rhs, v_rhs))

def fhn_euler_autograd(w):
    rhs = fhn_rhs_autograd(w)
    w_int = w + dt * rhs

    # Apply homogeneous Neumann boundary conditions
    w_new = np.concatenate(([w_int[1]], w_int[1:N-1], [w_int[N-2]], [w_int[N+1]], w_int[N+1:2*N-1], [w_int[2*N-2]]))

    return w_new

def fhn_euler_timestepper(w, T, verbose=False):
    N_steps = int(T / dt)
    for k in range(N_steps):
        if verbose:
            print('t =', k * dt)
        w = fhn_euler_autograd(w)
    return w

def psi(w, T):
    w_new = fhn_euler_timestepper(w, T)
    return w - w_new

def findSteadyState():
    # Initial condition
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 6.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 10, 0.0, 2.0, 0.1)
    w0 = np.concatenate((u0, v0))

    w0 = fhn_euler_timestepper(w0, 100.0)

    T = 1.0
    F = lambda w: psi(w, T)
    w = opt.newton_krylov(F, w0, f_tol=1.e-14, verbose=True)

    return w

def compareADWithFD():

    # Find the steady state
    print('Finding the steady state...')
    w_ss = findSteadyState()

    # Construct FD Jacobian
    T = 1.0
    eps_fd = 1.e-8
    dpsi_fd_ss = lambda q: (psi(w_ss + eps_fd * q, T) - psi(w_ss, T)) / eps_fd
    dpsi_fd = slg.LinearOperator((2*N, 2*N), matvec=dpsi_fd_ss)

    # Construct the AD Jacobian
    print('\nConstructing the Jacobians...')
    psi_T = lambda w: psi(w, T)
    dpsi_ad_ss = jacobian(psi_T)(w_ss)
    dpsi_ad = lambda q: np.dot(dpsi_ad_ss, q)

    # Compute the eigenvalues of both jacobians
    print('\nComputing the eigenvalues...')
    ad_eigvals = lg.eigvals(dpsi_ad)
    fd_eigvals = slg.eigvals(dpsi_fd, kind='LM', k=2*N-2)

    # Plot the eigenvalues
    plt.scatter(np.real(fd_eigvals), np.imag(fd_eigvals), color='tab:blue', label='Finite Differences')
    plt.scatter(np.real(ad_eigvals), np.imag(ad_eigvals), edgecolors='tab:orange', facecolors='none', label='Automatic Differentiation')
    plt.title(r'Eigevalues of $\nabla \psi$ at steady state')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    compareADWithFD()