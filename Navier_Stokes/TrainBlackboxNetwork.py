import torch as pt
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from BlackBoxDataset import NSDataSet
from BlackBoxModel import FeedforwardNetwork

# Just some sanity pytorch settings
pt.set_grad_enabled(True)
pt.set_default_dtype(pt.float64)

# Load the data in memory
print('Generating Training Data.')
batch_size = 300300
dataset = NSDataSet()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the Neural Network and the Optimizer (Adam)
print('\nSetting Up the Feed-Forward Neural Network.')
lr_step = 2500
network = FeedforwardNetwork()
loss_fn = nn.functional.mse_loss
optimizer = optim.Adam(network.parameters(), lr=0.001)
scheduler = sch.StepLR(optimizer, step_size=lr_step, gamma=0.1)
print('Number of Data Points per Trainable Parameter:', len(dataset) / network.n_trainable_parameters)

# Training Routine
train_losses = []
train_grads = []
train_counter = []
store_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/WaveData/'
def computeGradNorm():
    grads = []
    for param in network.parameters():
        grads.append(param.grad.view(-1))
    grads = pt.cat(grads)
    return pt.norm(grads)
def train(epoch):
    network.train()
    for _, (input_data, output_data) in enumerate(train_loader):
        optimizer.zero_grad()

        # Foward-propagate the input data
        network_output = network(input_data)

        # Compute the MSE Loss
        loss = loss_fn(network_output, output_data)

        # Compute loss gradient
        loss.backward()
        loss_grad = computeGradNorm()

        # Do one Adam optimization step
        optimizer.step()

    # Some housekeeping
    print('Train Epoch: {} \tLoss: {:.16f} \tLoss Gradient {:.16f}'.format(epoch, loss.item(), loss_grad.item()))
    train_losses.append(loss.item())
    train_grads.append(loss_grad.item())
    train_counter.append(epoch)
    pt.save(network.state_dict(), store_directory + 'model_black_box.pth')
    pt.save(optimizer.state_dict(), store_directory + 'optimizer_black_box.pth')

# Do the actual training
print('\nStarting Adam Training Procedure...')
n_epochs = 4 * lr_step
try:
    for epoch in range(1, n_epochs+1):
        train(epoch)
        scheduler.step()
except KeyboardInterrupt:
    print('Terminating Training. Plotting Training Error Convergence.')

# Show the training results
plt.semilogy(train_counter, train_grads, color='orange', label='Training Gradient')
plt.semilogy(train_counter, train_losses, color='blue', label='Training Loss')
plt.legend()
plt.xlabel('Epoch')
plt.show()