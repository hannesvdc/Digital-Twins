import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import BSpline

# An approximate bell function with mean 0.5 and variance 0.01
def test1():
    mu = 0.5
    sigma_sq = 0.1
    f = lambda q: 1.0 / np.sqrt(2.0*np.pi*sigma_sq) * np.exp(-( q - mu)**2 / (2.0*sigma_sq))
    x_values = np.linspace(0.0, 1.0, 20)
    f_values = f(x_values)
    spline = BSpline.ClampedCubicSpline(x_values, f_values, solver='krylov')

    plot_x_values = np.linspace(0.0, 1.0, 1000)
    spline_values = spline(plot_x_values)

    plt.plot(plot_x_values, f(plot_x_values), label='Original function')
    plt.plot(plot_x_values, spline_values, linestyle='dashed', label='Spline Interpolation')
    plt.scatter(x_values, f_values,  color='green', label='Interpolation Points')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test1()