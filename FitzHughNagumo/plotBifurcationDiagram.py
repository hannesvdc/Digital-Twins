import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

N = 200
L = 20.0
dx = L / N
dt = 0.001
T = 1.0

def plotBifurcationDiagram():
    M = 2 * N

    # Load Continuation data, Arnoldi and QR data
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    bf_data = np.load(directory + 'bf_diagram.npy')
    arnoldi_values = np.load(directory + 'Arnoldi_Eigenvalues.npy')
    qr_values = np.load(directory + 'QR_Eigenvalues.npy')
    arnoldi_scipy_values = np.load(directory + 'Arnoldi_Scipy_Eigenvalues.npy')

    n_steps = bf_data.shape[0]
    plot_x1_path = np.average(bf_data[:,0:N], axis=1)
    eps1_path = bf_data[:, M]
    plot_x2_path = np.average(bf_data[:,M+1 : M+1+N], axis=1)
    eps2_path = bf_data[:, -1]

    # Plot both branches
    plt.plot(eps1_path, plot_x1_path, color='blue', label='Branch 1')
    plt.plot(eps2_path, plot_x2_path, color='red', label='Branch 2')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$<u>$')

    plt.figure()
    plt.plot(np.linspace(0, n_steps-1, n_steps), np.real(arnoldi_values[0,:]), color='blue', label='Arnoldi Eigenvalues')
    plt.plot(np.linspace(0, n_steps-1, n_steps), np.real(arnoldi_scipy_values[0,:]), color='purple', label='Arnoldi Scipy Eigenvalues')
    plt.plot(np.linspace(0, n_steps-1, n_steps), np.real(qr_values[0,:]), color='black', label='QR Eigenvalues (Exact)')
    plt.title('Branch 1')
    plt.xlabel('Continuation Step')
    plt.ylabel('Eigenvalue')
    plt.legend()
    
    plt.figure()
    plt.scatter(np.real(arnoldi_values[1,:]), np.abs(np.imag(arnoldi_values)[1,:]), color='red', label='Arnoldi Eigenvalues')
    plt.plot(np.linspace(0, n_steps-1, n_steps), np.real(arnoldi_scipy_values[1,:]), color='purple', label='Arnoldi Scipy Eigenvalues')
    plt.scatter(np.real(qr_values[1,:]), np.abs(np.imag(qr_values[1,:])), color='black', label='QR Eigenvalues (Exact)')
    plt.title('Branch 2')
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.legend()
    plt.show()

