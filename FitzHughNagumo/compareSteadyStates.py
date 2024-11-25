import numpy as np
import matplotlib.pyplot as plt

directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'

print('\nLoading Steady-State of the Euler Scheme ...')
euler_data = np.load(directory + 'euler_steady_state.npy')
x_euler = euler_data[0,:]
U_euler = euler_data[1,:]
V_euler = euler_data[2,:]

print('\nLoading Steady-State of the Tooth-No-Gap Scheme')
gap_no_tooth_data = np.load(directory + 'tooth_no_gap_steady_state.npy')
x_tooth = gap_no_tooth_data[0,:]
U_tooth = gap_no_tooth_data[1,:]
V_tooth = gap_no_tooth_data[2,:]

print('\nLoading Steady-State of the Gap-Tooth Scheme')
n_teeth = 100
n_points_per_tooth = 11
gap_tooth_data = np.load(directory + 'gaptooth_steady_state.npy')
x_gaptooth = gap_tooth_data[0,:]
U_ss_gaptooth = gap_tooth_data[1,:]
V_ss_gaptooth = gap_tooth_data[2,:]
x_plot_array = []
u_patch_gt = []
v_patch_gt = []
for i in range(n_teeth):
    x_plot_array.append(x_gaptooth[i * n_points_per_tooth : (i+1) * n_points_per_tooth])
    u_patch_gt.append(U_ss_gaptooth[i * n_points_per_tooth : (i+1) * n_points_per_tooth])
    v_patch_gt.append(V_ss_gaptooth[i * n_points_per_tooth : (i+1) * n_points_per_tooth])

plt.plot(x_plot_array[0], u_patch_gt[0], label=r'Gap-Tooth $u(x)$', color='tab:blue')
plt.plot(x_plot_array[0], v_patch_gt[0], label=r'Gap-Tooth $v(x)$', color='tab:orange')
for i in range(1, n_teeth):
    plt.plot(x_plot_array[i], u_patch_gt[i], color='tab:blue')
    plt.plot(x_plot_array[i], v_patch_gt[i], color='tab:orange')
plt.plot(x_euler, U_euler, label=r'Euler $u(x)$', color='tab:green')
plt.plot(x_euler, V_euler, label=r'Euler $v(x)$', color='tab:red')
plt.plot(x_tooth, U_tooth, label=r'Tooth-No-Gap $u(x)$', color='tab:purple')
plt.plot(x_tooth, V_tooth, label=r'Tooth-No-Gap $v(x)$', color='tab:brown')
plt.xlabel(r'$x$')
plt.ylabel(r'$u, v$', rotation=0)
plt.title('Newton-Krylov Steady-State')
plt.legend()
plt.show()