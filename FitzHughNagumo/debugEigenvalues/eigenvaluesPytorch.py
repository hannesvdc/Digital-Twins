import torch as pt
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

pt.set_default_dtype(pt.float64)
pt.set_grad_enabled(True)

# Model parameters
L = 20.0
N = 200
dx = L / N
a0 = -0.03
a1 = 2.0
delta = 4.0
eps = 0.1 # 0.01 originally for the spatio-temporal oscillations
params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

# Pytorch Implementations of Timnestepper Functions
def fhn_rhs(u, v):
    u_left = pt.roll(u, -1)
    u_right = pt.roll(u, 1)
    v_left = pt.roll(v, -1)
    v_right = pt.roll(v, 1)

    u_xx = (u_left - 2.0*u + u_right) / dx**2
    v_xx = (v_left - 2.0*v + v_right) / dx**2

    u_rhs = u_xx + u - u**3 - v
    v_rhs = params['delta'] * v_xx + params['eps'] * (u - params['a1']*v - params['a0'])

    return u_rhs, v_rhs

def fhn_euler(u, v, dt):
    u_rhs, v_rhs = fhn_rhs(u, v)
    u_new = u + dt * u_rhs
    v_new = v + dt * v_rhs

    # Apply homogeneous Neumann boundary conditions
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    v_new[0] = v_new[1]
    v_new[-1] = v_new[-2]

    return u_new, v_new

def fhn_euler_timestepper(w, dt, T, verbose=False):
    u = w[:N]
    v = w[N:]

    N_steps = int(T / dt)
    for k in range(N_steps):
        if verbose:
            print('t =', k * dt)
        u, v = fhn_euler(u, v, dt)
    return pt.concatenate((u, v))

def psi(w, T, dt):
    w_new = fhn_euler_timestepper(w, dt, T)
    return w - w_new

def plotADEigenvalues():
    T = 1.0
    dt = 1.e-3
    
    # Load the steady state solution from file (using numpy)
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    steady_state_filename = 'euler_steady_state.npy'
    steady_state = np.load(directory + steady_state_filename)
    w_ss = pt.concatenate((pt.tensor(steady_state[1,:]), pt.tensor(steady_state[2,:])))

    # Calculate psi-value in the steady state
    input = pt.clone(w_ss).requires_grad_(True)
    output = psi(input, T, dt)

    # Setup the Analytic Jacobian Matrix
    dF_ad = np.zeros((2*N, 2*N))
    for n in range(2*N):
        grad_output = pt.zeros_like(output)
        grad_output[n] = 1.0  # Select the i-th component

        grad_n = pt.autograd.grad(outputs=output, inputs=input, grad_outputs=grad_output, retain_graph=True)[0]
        dF_ad[n,:] = grad_n.detach().numpy()

    # Calculate the Finite Difference Jacobian Matrix
    eps_fd = 1.e-8
    dF_fd = np.zeros((2*N, 2*N))
    def df_fd(v):
        _v = pt.tensor(v)
        df = (psi(w_ss + eps_fd * _v, T, dt) - psi(w_ss, T, dt)) / eps_fd
        return df.detach().numpy()
    for n in range(2*N):
        e_n = np.eye(2*N)[:,n]
        grad_n = df_fd(e_n)
        dF_fd[n,:] = grad_n

    # Calculate the eigenvalues of these numpy matrices
    eigvals_ad, eigvecs_ad = lg.eig(dF_ad)
    eigvals_fd, eigvecs_df = lg.eig(dF_fd)

    # Plot the eigenvalues
    plt.scatter(np.real(eigvals_ad), np.imag(eigvals_ad), label='Automatic Differentiation')
    plt.scatter(np.real(eigvals_fd), np.imag(eigvals_fd), edgecolors='tab:orange', facecolor='none', label='Finite Differences')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(r'Eigenvalues of the Jacobian of $\psi$')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plotADEigenvalues()
