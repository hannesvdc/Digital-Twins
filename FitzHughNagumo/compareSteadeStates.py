import matplotlib.pyplot as plt

from  EulerTimestepper import findSteadyState as calc_ss_euler
from ToothNogapTimestepper import findSteadyStateNewtonGMRES as calc_ss_tooth

x_euler, ss_euler = calc_ss_euler(return_ss=True)
x_tooth, ss_tooth = calc_ss_tooth(return_ss=True)
print(ss_tooth)

# Extract U and V
N_euler = len(ss_euler) // 2
U_euler, V_euler = ss_euler[0:N_euler], ss_euler[N_euler:]
N_tooth = len(ss_tooth) // 2
U_tooth, V_tooth = ss_tooth[0:N_tooth], ss_tooth[N_tooth:]

plt.plot(x_euler, U_euler, label=r'Euler $u(x, t=\infty)$')
plt.plot(x_euler, V_euler, label=r'Euler $v(x, t=\infty)$')
plt.plot(x_tooth, U_tooth, label=r'Tooth $u(x, t=\infty)$')
plt.plot(x_tooth, V_tooth, label=r'Tooth $v(x, t=\infty)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$u, v$', rotation=0)
plt.legend()
plt.show()