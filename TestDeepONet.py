import numpy as np
import matplotlib.pyplot as plt

from api.DeepONet import DeepONet
from api.Functional import sigmoid

N = 100
scale = 1.e9

# Load the Training Data (should be separate test data later on)
directory = '/Users/hannesvdc/Research_Data/Digital Twins/Elastodynamics/'
f_filename = 'branch_data.npy'
y_filename = 'trunk_data.npy'
output_filename = 'output_data.npy'
branch_data = np.load(directory + f_filename)
trunk_data  = np.load(directory + y_filename)
output_data = np.load(directory + output_filename)
print('outpput', output_data)

# Load weights for deeponet
directory = '/Users/hannesvdc/Research_Data/Digital Twins/Elastodynamics/'
filename = 'Weights_Adam.npy'
weights = np.load(directory + filename)

# Setup the DeepONet
p = 25
branch_layers = [202, p, p]
trunk_layers = [2, p, p]
deeponet = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers, activation=sigmoid)
print('Number of weights:', deeponet.n_weights)

# Pick a random forcing f (branch_data[:,i]) and compute output of DeepONet for all pairs (x, y) in trunk_data
index = 0
f = branch_data[:,[index]]
G = deeponet.forward(f, trunk_data, weights)
print(G.shape)

# Cast output G into a matrix of u
u = np.zeros((N+1, N+1))
for counter in range(G.shape[1]):
    i = counter // (N+1)
    j = counter %  (N+1)
    u[j,i] = G[0,counter] / scale
print('u(x, y)', u, np.min(u), np.min(f))

# Plot the results
min_u = np.min(u)
max_u = np.max(u)
print('\nPlotting...')
x = np.linspace(0.0, 1.0, N+1)
y = np.linspace(0.0, 1.0, N+1)
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X, Y, u, shading='auto', vmin=min_u, vmax=max_u, cmap='jet')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$u(x,y)$')
plt.colorbar()
plt.show()
