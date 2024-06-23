import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from DeepONet import DeepONet

# The data impleementation and loader class
class DeepONetDataset(Dataset):
    def __init__(self):
        super().__init__()
        print('Loading Data...')
        
        directory = '/Users/hannesvdc/Research_Data/Digital Twins/DeepONet/'
        branch_filename = 'branch_data.npy'
        trunk_filename  = 'trunk_data.npy'
        G_filename      = 'output_data.npy'

        branch_data = pt.from_numpy(np.load(directory + branch_filename).transpose()) # Transpose to make data row-major
        n_branch_points = branch_data.shape[0]
        branch__el_size = branch_data.shape[1]
        trunk_data = pt.from_numpy(np.load(directory + trunk_filename).transpose())
        n_trunk_points = trunk_data.shape[0]
        trunk_el_size = trunk_data.shape[1]
        loaded_output_data = pt.from_numpy(np.load(directory + G_filename))
        loaded_output_size = np.prod(loaded_output_data.shape)

        print('Structuring Data...')
        scale = 1.e9
        self.input_data = pt.zeros((n_branch_points * n_trunk_points, branch__el_size + trunk_el_size))
        self.output_data = pt.zeros(loaded_output_size)
        for n in range(self.input_data.shape[0]):
            i = n // n_trunk_points
            j = n % n_trunk_points
            self.input_data[n,:] = pt.cat((branch_data[i,:], trunk_data[j,:]))
            self.output_data[n] = scale * loaded_output_data[i,j]

        # Downsampling
        print('Downsampling...')
        n_samples = 200 * 256
        indices = np.random.randint(low=0, high=loaded_output_size, size=n_samples)
        self.input_data = self.input_data[indices,:]
        self.output_data = self.output_data[indices]

    def __len__(self):
        return self.input_data.shape[0]
	
    def __getitem__(self, idx):
        return self.input_data[idx,:], self.output_data[idx]
    
# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Load the data in memory
batch_size = 256
dataset = DeepONetDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Network and the Optimizer (Adam)
print('\nSetting Up DeepONet Neural Net...')
p = 25
branch_layers = [202, p, p]
trunk_layers = [2, p, p]
network = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers)
network.forward(dataset.input_data)
optimizer = optim.Adam(network.parameters())

# Training Routine
loss_fn = nn.MSELoss()
train_losses = []
train_counter = []
log_rate = 100
store_directory = '/Users/hannesvdc/Research_Data/Digital Twins/DeepONet/results/'
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Compute Loss
        output = network(data)
        loss = loss_fn(output, target)

        # Compute loss gradient and do one optimization step
        loss.backward()
        optimizer.step()

        # Some housekeeping
        if batch_idx % log_rate == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

            pt.save(network.state_dict(), store_directory + 'model.pth')
            pt.save(optimizer.state_dict(), store_directory + 'optimizer.pth')

# Do the actual training
print('\nStarting Training Procedure...')
n_epochs = 50000
try:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
except KeyboardInterrupt:
    print('Aborting Training. Plotting Training Convergence.')

# Show the training results
fig = plt.figure()
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Number of training examples seen')
plt.ylabel('Loss')
plt.show()