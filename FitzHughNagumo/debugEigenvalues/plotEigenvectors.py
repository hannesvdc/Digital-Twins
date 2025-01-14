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
    ss_filename = 'euler_steady_state.npy'
    return np.load(directory + ss_filename)

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

    # Calculate the leading eigenvalues of the Jacobian of f. 
    # Leading = smallest magnitude (closest to zero)
    eigvals, eigvects = slg.eigs(Df, k=2*N-2, which='SM', return_eigenvectors=True)
    print(eigvals)


if __name__ == '__main__':
    plotDfEigenvectors()