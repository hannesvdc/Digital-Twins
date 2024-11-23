import numpy as np
import matplotlib.pyplot as plt

from GapToothTimestepper import findSteadyStateNewtonGMRES as calc_ss_gaptooth

directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'

print('\nLoading Steady-State of the Euler Scheme ...')
euler_data = np.load(directory + 'euler_steady_state.npy')
x_euler = euler_data[0,:]
U_euler = euler_data[1,:]
V_euler = euler_data[2,:]

print('\nCalculating Steady-State of the Tooth-No-Gap Scheme')
gap_no_tooth_data = np.load(directory + 'gap_no_tooth_steady_state.npy')
x_tooth = gap_no_tooth_data[0,:]
U_tooth = gap_no_tooth_data[1,:]
V_tooth = gap_no_tooth_data[2,:]

print('\nCalculating Steady-State of the Gap-Tooth Scheme')
x_gaptooth, U_ss_gaptooth, V_ss_gaptooth = calc_ss_gaptooth(return_ss=True)

for i in range(len(x_gaptooth)):
    if i == 0:
        plt.plot(x_gaptooth[i], U_ss_gaptooth[i], label=r'Gap-Tooth $u(x)$', color='tab:blue')
        plt.plot(x_gaptooth[i], V_ss_gaptooth[i], label=r'Gap-Tooth $v(x)$', color='tab:orange')
    else:
        plt.plot(x_gaptooth[i], U_ss_gaptooth[i], color='tab:blue')
        plt.plot(x_gaptooth[i], V_ss_gaptooth[i], color='tab:orange')
plt.plot(x_euler, U_euler, label=r'Euler $u(x)$', color='tab:green')
plt.plot(x_euler, V_euler, label=r'Euler $v(x)$', color='tab:red')
plt.plot(x_tooth, U_tooth, label=r'Tooth-No-Gap $u(x)$', color='tab:purple')
plt.plot(x_tooth, V_tooth, label=r'Tooth-No-Gap $v(x)$', color='tab:brown')
plt.xlabel(r'$x$')
plt.ylabel(r'$u, v$', rotation=0)
plt.title('Newton-Krylov Steady-State')
plt.legend()
plt.show()