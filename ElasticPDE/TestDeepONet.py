import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from DeepONet import DeepONet
import ElasticPDE as pde

# The data impleementation and loader class
class DeepONetDataset(Dataset):
    def __init__(self):
        super().__init__()
        print('Loading Data...')
        self.N = 100
        
        directory = '/Users/hannesvdc/Research_Data/Digital Twins/DeepONet/'
        branch_filename = 'branch_data.npy'
        trunk_filename  = 'trunk_data.npy'

        branch_data = pt.from_numpy(np.load(directory + branch_filename).transpose()) # Transpose to make data row-major
        branch__el_size = branch_data.shape[1]
        trunk_data = pt.from_numpy(np.load(directory + trunk_filename).transpose())
        n_trunk_points = trunk_data.shape[0]
        trunk_el_size = trunk_data.shape[1]

        print('Structuring Data...')
        self.scale = 1.e9
        self.input_data = pt.zeros((n_trunk_points, branch__el_size + trunk_el_size))
        for n in range(self.input_data.shape[0]):
            self.input_data[n,:] = pt.cat((branch_data[0,:], trunk_data[n,:]))

    def __len__(self):
        return self.input_data.shape[0]
	
    def __getitem__(self, idx):
        return self.input_data[idx,:]
    
# No need for gradients in test script
pt.set_grad_enabled(False)
pt.set_default_dtype(pt.float64)

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up DeepONet Neural Net...')
p = 25
branch_layers = [202, p, p]
trunk_layers = [2, p, p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)

# Load the network form data
store_directory = '/Users/hannesvdc/Research_Data/Digital Twins/DeepONet/results/'
filename = 'model_p=25.pth'
network.load_state_dict(pt.load(store_directory + filename))
network.eval()

# Evaluate network
dataset = DeepONetDataset()
N = dataset.N
G_NN = network.forward(dataset.input_data)
G_NN = pt.reshape(G_NN, (N+1, N+1)).T / dataset.scale

# Solve the Elastostatic PDE with given forcing
print('\nComputing Solution to Elastostatic PDE.')
E = 410.0 * 1.e3
mu = 0.3
forcing = np.zeros((N+1, 2))
forcing[:,0] = dataset.input_data[0,0:(N+1)]
forcing[:,1] = dataset.input_data[0, (N+1) : 2*(N+1)]
A = pde.computeSystemMatrix(mu, N)
lu, pivot = sc.linalg.lu_factor(A)
(u, v) = pde.solveElasticPDE(lu, pivot, E, mu, forcing, dataset.N)

# Plot result
x = np.linspace(0.0, 1.0, N+1)
y = np.linspace(0.0, 1.0, N+1)
X, Y = np.meshgrid(x, y)
v_min = min(np.min(u), pt.min(G_NN))
v_max = max(np.max(u), pt.max(G_NN))

plt.pcolormesh(X, Y, u, shading='auto', cmap='jet', vmin=v_min, vmax=v_max)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Reference Displacement $u(x,y)$')
plt.colorbar()

plt.figure()
plt.pcolormesh(X, Y, G_NN, shading='auto', cmap='jet', vmin=v_min, vmax=v_max)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'DeepONet Displacement $u(x,y)$')
plt.colorbar()
plt.show()