import matplotlib.pyplot as plt

from  EulerTimestepper import findSteadyState as calc_ss_euler
from ToothNogapTimestepper import findSteadyStateNewtonGMRES as calc_ss_tooth
from GapToothTimestepper import findSteadyStateNewtonGMRES as calc_ss_gaptooth

print('\nCalculating Steady-State of the Euler Scheme')
x_euler, ss_euler = calc_ss_euler(return_ss=True)

print('\nCalculating Steady-State of the Tooth-No-Gap Scheme')
x_tooth, ss_tooth = calc_ss_tooth(return_ss=True)

print('\nCalculating Steady-State of the Gap-Tooth Scheme')
x_gaptooth, U_ss_gaptooth, V_ss_gaptooth = calc_ss_gaptooth(return_ss=True)

# Extract U and V for euler and the tooth-scheme
N_euler = len(ss_euler) // 2
U_euler, V_euler = ss_euler[0:N_euler], ss_euler[N_euler:]
N_tooth = len(ss_tooth) // 2
U_tooth, V_tooth = ss_tooth[0:N_tooth], ss_tooth[N_tooth:]

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