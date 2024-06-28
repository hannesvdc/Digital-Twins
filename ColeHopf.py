import numpy as np
import numpy.fft as fft

# We assume grid ranges from 0 to L
def heatToBurgers(grid, u, D):
    assert len(grid) == len(u)

    # Compute the derivative of u to x (grid) using the Fourier transform
    M = len(grid)
    L = grid[M-1]
    k = np.concatenate((np.arange(M / 2 + 1), np.arange(-M / 2 + 1, 0))) * 2.0 * np.pi / L # DC at 0, k > 0 first, then f < 0
    F_u = fft.fft(u)
    F_ux = 1j * k * F_u
    u_x = np.real(fft.ifft(F_ux))

    # Compute the Cole-Hopf Transform
    v = 2.0 * D * u_x / u
    return v