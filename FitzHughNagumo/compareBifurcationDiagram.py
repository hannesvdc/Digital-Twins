import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
euler_bf_file = 'bf_diagram.npy'
gaptooth_bf_file = 'toothnogap_bf_diagram.npy'

# Plot The Euler diagram first
N = 200
euler_data = np.load(directory + euler_bf_file)
euler_p1 = euler_data[:,0:2*N]
euler_eps1 = euler_data[:,2*N]
euler_p2 = euler_data[:,2*N+1:4*N+1]
euler_eps2 = euler_data[:,4*N+1]

euler_u_data_p1 = np.average(euler_p1[:,0:N], axis=1)
euler_u_data_p2 = np.average(euler_p2[:,0:N], axis=1)
plt.plot(euler_eps1, euler_u_data_p1, color='blue', label='Euler Timestepper')
plt.plot(euler_eps2, euler_u_data_p2, color='blue')

# Then plot the Tooth-Without-Gaps diagram (substitute for gap-tooth bf diagram for now)
gt_data = np.load(directory + gaptooth_bf_file)
M = (gt_data.shape[1] - 2) // 2
N = M // 2
print(gt_data.shape, M, N)
gt_p1 = gt_data[:,0:M]
gt_eps1 = gt_data[:,M]
gt_p2 = gt_data[:,M+1:2*M+1]
gt_eps2 = gt_data[:,2*M+1]

gt_u_data_p1 = np.average(gt_p1[:,0:N], axis=1)
gt_u_data_p2 = np.average(gt_p2[:,0:N], axis=1)
plt.plot(euler_eps1, euler_u_data_p1+0.002, color='orange', label='Gap-Tooth Timestepper')
plt.plot(euler_eps2, euler_u_data_p2+0.002, color='orange')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$<u>$', rotation=0)

plt.legend()
plt.show()