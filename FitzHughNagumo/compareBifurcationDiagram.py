import numpy as np
import matplotlib.pyplot as plt

directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
euler_bf_file = 'euler_bf_diagram.npy'
tooth_no_gap_bf_file = 'tooth_no_gap_bf_diagram.npy'
gaptooth_bf_file = 'gaptooth_bf_diagram.npy'

is_sorted = lambda a: np.all(a[:-1] <= a[1:])

# Plot The Euler diagram first
N = 200
M = 2*N
euler_data = np.load(directory + euler_bf_file)
euler_p1 = euler_data[:,0:M]
euler_eps1 = euler_data[:,M]
euler_p2 = euler_data[:,M+1:2*M+1]
euler_eps2 = euler_data[:,2*M+1]
euler_u_data_p1 = np.average(euler_p1[:,0:N], axis=1)
euler_u_data_p2 = np.average(euler_p2[:,0:N], axis=1)

# Find the location of the Hopf Bifurcation
eps_hopf = 0.015
p = euler_eps2.argsort()
euuler_eps2_sorted = euler_eps2[p]
euler_u_data_p2_sorted = euler_u_data_p2[p]
hopf_index = np.searchsorted(euuler_eps2_sorted, eps_hopf)
u_hopf = euler_u_data_p2_sorted[hopf_index]

plt.plot(euler_eps1, euler_u_data_p1, color='tab:blue', label='Euler Timestepper')
plt.plot(euler_eps2, euler_u_data_p2, color='tab:blue')

# Then plot the Tooth-Without-Gaps diagram (substitute for gap-tooth bf diagram for now)
tng_data = np.load(directory + tooth_no_gap_bf_file)
M = (tng_data.shape[1] - 2) // 2
N = M // 2
tng_p1 = tng_data[:,0:M]
tng_eps1 = tng_data[:,M]
tng_p2 = tng_data[:,M+1:2*M+1]
tng_eps2 = tng_data[:,2*M+1]
tng_u_data_p1 = np.average(tng_p1[:,0:N], axis=1)
tng_u_data_p2 = np.average(tng_p2[:,0:N], axis=1)
plt.plot(tng_eps1, tng_u_data_p1, color='tab:orange', label='Tooth-Without-Gaps Timestepper')
plt.plot(tng_eps2, tng_u_data_p2, color='tab:orange')

# Finally plot the Gap-Tooth bifurcation diagram
gt_data = np.load(directory + gaptooth_bf_file)
M = (gt_data.shape[1] - 2) // 2
N = M // 2
gt_p1 = gt_data[:,0:M]
gt_eps1 = gt_data[:,M]
gt_p2 = gt_data[:,M+1:2*M+1]
gt_eps2 = gt_data[:,2*M+1]

gt_u_data_p1 = np.average(gt_p1[:,0:N], axis=1)
gt_u_data_p2 = np.average(gt_p2[:,0:N], axis=1)
plt.plot(gt_eps1, gt_u_data_p1, color='tab:green', label='Gap-Tooth Timestepper')
plt.plot(gt_eps2, gt_u_data_p2, color='tab:green')
plt.xlabel(r'$\varepsilon$')
plt.ylabel(r'$<u>$', rotation=0)

plt.scatter([eps_hopf], [u_hopf], color='red', label='Hopf Bifurcation')

plt.legend()
plt.show()