import torch as pt
import torch.fft as fft

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from BlackBoxDataset import NSDataSet
from BlackBoxModel import FeedforwardNetwork

# Just some sanity pytorch settings
pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

def calcSpatialDerivatives(y, k):
    f = fft.fft(y)
    y_x = pt.real(fft.ifft( 1j * k * f ))
    y_xx = pt.real(fft.ifft( (1j * k)**2 * f ))
    y_xxx = pt.real(fft.ifft( (1j * k)**3 * f ))
    y_xxxx = pt.real(fft.ifft( (1j * k)**4 * f ))
    
    return y_x, y_xx, y_xxx, y_xxxx

def rhs(network, y, k):
    y_x, y_xx, y_xxx, y_xxxx = calcSpatialDerivatives(y, k)
    input_data = pt.vstack((y, y_x, y_xx, y_xxx, y_xxxx)).transpose(0, 1)

    return network.forward(input_data)[:,0]

def RK4(network, y, k, dt):
    k1 = rhs(network, y, k)
    k2 = rhs(network, y + 0.5*dt*k1, k)
    k3 = rhs(network, y + 0.5*dt*k2, k)
    k4 = rhs(network, y + 1.0*dt*k3, k)

    return y + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

# Load the initial condition
print('Loading Initial Condition.')
storage_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/WaveData/'
y_filename = 'newRe1p75_y.dat'
initial_data = pt.from_numpy(np.loadtxt(storage_directory + y_filename)[:,0])
M = initial_data.size()[0]
L = 95.0
k = pt.from_numpy(np.concatenate((np.arange(M // 2 + 1), np.arange(-M // 2 + 1, 0))) * 2.0 * np.pi / L)

# Load the optimized network state
network = FeedforwardNetwork()
network.load_state_dict(pt.load(storage_directory + 'model_black_box.pth'))

# Simulate the KS equations and store the intermediate solutions
time_simulation = [initial_data]
y = pt.clone(initial_data)
dt = 0.01
T = 1.0
N = int(T / dt)
for n in range(N):
    print('t =', (n+1)*dt)
    y = RK4(network, y, k, dt)
    time_simulation.append(y)
