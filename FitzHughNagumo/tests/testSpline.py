import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

import BSpline

# An approximate bell function with mean 0.5 and variance 0.1
def test1():
    mu = 0.5
    sigma_sq = 0.01
    x_values = np.linspace(0.0, 1.0, 20)
    f_values = 1.0 / np.sqrt(2.0*np.pi*sigma_sq) * np.exp(-(x_values - mu)**2 / (2.0*sigma_sq))
    spline = BSpline.ClampedCubicSpline(x_values, f_values)

    plot_x_values = np.linspace(0.0, 1.0, 1000)
    spline_values = np.zeros_like(plot_x_values)
    for i in range(len(plot_x_values)):
        spline_values[i] = spline(plot_x_values[i])

    plt.plot(x_values, f_values+0.01, label='Original function')
    plt.plot(plot_x_values, spline_values, label='Spline Interpolation')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test1()