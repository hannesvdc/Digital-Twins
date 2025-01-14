import sys
sys.path.append('../')

import numpy as np
import numpy.linalg as lg
import scipy.optimize as opt
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

# Import all f functions from EulerTimestepper.py
from EulerTimestepper import f_wrapper

# Shared model parameters
L = 20.0
N = 200
dx = L / N
a0 = -0.03
a1 = 2.0
delta = 4.0
eps = 0.1 # 0.01 originally for the spatio-temporal oscillations
params = {'delta': delta, 'eps': eps, 'a0': a0, 'a1': a1}

def fetchSteadyState():
    """ Calculate the steady-state of the PDE for a fixed value of epsilon. 
        For now, we just load it from file because it is available.
    """
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    steady_state_filename = 'euler_steady_state.npy'
    steady_state = np.load(directory + steady_state_filename)

    return np.concatenate((steady_state[1,:], steady_state[2,:]))

def plotDfEigenvectors():
    eps_fd = 1.e-8

    # Finite differences approximation of the Jacobian of f (the PDE right-hand side)
    w_inf = fetchSteadyState()
    def df(v):
        """ w_inf = the steady state solution of the PDE
            v = the eigenvector of the Jacobian of f
        """
        return (f_wrapper(w_inf + eps_fd * v, N, dx, params) - f_wrapper(w_inf, N, dx, params)) / eps_fd

    # Setup the Jacobian matrix of f using df
    Df = slg.LinearOperator(shape=(2*N, 2*N), matvec=df)
    Df_matrix = np.zeros(Df.shape)
    for n in range(2*N):
        e_n = np.eye(2*N)[:,n]
        Df_matrix[:,n] = df(e_n)

    # Calculate all eigenvalues of the Jacobian of f. 
    eigvals, eigvecs = lg.eig(Df_matrix)

    # Sort the eigenvalue / eigenvector pairs by the real part of the eigenvalue
    real_parts = np.real(eigvals)
    indices = np.argsort(real_parts)[::-1]
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:,indices]
    print(eigvals)

    # Plot the eigenvalues
    fig, ax = plt.subplots()
    ax.scatter(eigvals.real, eigvals.imag, label=r'Eigenvalues $\lambda$ of $\nabla f$')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.legend()

    # Plot the steady-state and the eigenvectors
    x_arrays = np.linspace(0, L, N)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_arrays, w_inf[:N], label=r'$u_{\infty}$')
    ax2.plot(x_arrays, w_inf[N:], label=r'$v_{\infty}$')
    for n in range(10):
        ax1.plot(x_arrays, eigvecs[:N,n].real)
        ax2.plot(x_arrays, eigvecs[N:,n].real, label=f'$v_{n}$')
    ax1.set_xlabel(r'$x$')
    ax2.set_xlabel(r'$x$')
    ax1.legend()
    ax2.legend()
    plt.show()
    

if __name__ == '__main__':
    plotDfEigenvectors()