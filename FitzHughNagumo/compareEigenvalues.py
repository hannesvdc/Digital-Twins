import numpy as np
import matplotlib.pyplot as plt

directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
euler_eigvals = np.load(directory + 'euler_eigenvalues.npy')
toothnogap_eigvals = np.load(directory + 'tooth_no_gap_eigenvalues.npy')
gaptooth_eigvals = np.load(directory + 'gaptooth_eigenvalues.npy')

# Load components
euler_psi = euler_eigvals[0,:]
euler_f = euler_eigvals[1,:]
euler_psi_approx = euler_eigvals[2,:]

# Plot in two separate figures
plt.scatter(np.real(euler_psi_approx), np.imag(euler_psi_approx), label=r'$1 - \exp(\sigma T)$')
plt.scatter(np.real(euler_psi), np.imag(euler_psi), label='Euler Timestepper')
plt.scatter(np.real(toothnogap_eigvals), np.imag(toothnogap_eigvals), label='Tooth-No-Gap Timestepper')
plt.scatter(np.real(gaptooth_eigvals), np.imag(gaptooth_eigvals), label='Gap-Tooth Timestepper')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(visible=True, which='major', axis='both')
plt.title('Timestepper Eigenvalues')
plt.legend()

plt.figure()
plt.scatter(np.real(euler_f), np.imag(euler_f), label=r'PDE Eigenvalues $\sigma$')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(visible=True, which='major', axis='both')
plt.title('PDE Eigenvalues')
plt.legend()
plt.show()