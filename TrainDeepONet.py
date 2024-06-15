import jax.numpy as np
import jax.numpy.linalg as lg
import numpy.random as rd
from jax import grad, jit
import matplotlib.pyplot as plt

from api.DeepONet import DeepONet
from api.Functional import sigmoid
from api.Adam import AdamOptimizer
from api.Scheduler import PiecewiseConstantScheduler

# Load the Training Data
directory = '/Users/hannesvdc/Research_Data/Digital Twins/Elastodynamics/'
f_filename = 'branch_data.npy'
y_filename = 'trunk_data.npy'
g_filename = 'output_data.npy'
branch_data = np.load(directory + f_filename)
trunk_data  = np.load(directory + y_filename)
G = np.load(directory + g_filename)
print('Maximal G', np.max(G), 0.01**2 * (1-0.3**2) / (410 * 10**3))

# Setup the DeepONet
p = 25
branch_layers = [202, p, p]
trunk_layers = [2, p, p]
deeponet = DeepONet(branch_layers=branch_layers, trunk_layers=trunk_layers, activation=sigmoid)
print('Number of weights:', deeponet.n_weights)

# Setup the loss function and its gradient for learning
scale = 1.e9
stable_norm_sq = jit(lambda x: np.sum(np.square(x)))
loss_fn = jit(lambda weights: stable_norm_sq(deeponet.forward(branch_data, trunk_data, weights) - scale * G) / G.size)
d_loss_fn = jit(grad(loss_fn))

# Sample the Initial Weights and compute Initial Loss & Gradient
rng = rd.RandomState()
weights = rng.normal(0.0, 1.0, size=deeponet.n_weights)
loss = loss_fn(weights)
while loss > 1.e3:
    weights = rng.normal(0.0, 1.0, size=deeponet.n_weights)
    loss = loss_fn(weights)
print('Initial Weights', weights)
print('Initial Loss', loss)
print('Initial Loss Gradient', lg.norm(d_loss_fn(weights)))

# Setup the Optimizer and Learn Optimal Weights
epochs = 50000
scheduler = PiecewiseConstantScheduler({0: 1.e-2, 45000: 1.e-3})
optimizer = AdamOptimizer(loss_fn, d_loss_fn, scheduler=scheduler)
try:
    weights = optimizer.optimize(weights, n_epochs=epochs, tolerance=0.0)
except KeyboardInterrupt: # If Training has converged well enough with Adam, the user can stop manually
    print('Aborting Training. Plotting Convergence')
print('Done Training at', len(optimizer.losses), 'epochs. Weights =', weights)
losses = np.array(optimizer.losses)
grad_norms = np.array(optimizer.gradient_norms)

# Storing weights
directory = '/Users/hannesvdc/Research_Data/Digital Twins/Elastodynamics/'
filename = 'Weights_Adam.npy'
np.save(directory + filename, weights)

# Post-processing
x_axis = np.arange(len(losses))
plt.grid(linestyle = '--', linewidth = 0.5)
plt.semilogy(x_axis, losses, label='Training Loss')
plt.semilogy(x_axis, grad_norms, label='Loss Gradient', alpha=0.3)
plt.xlabel('Epoch')
plt.suptitle('DeepONet Training Convergence')
plt.title(r'Adam')
plt.legend()
plt.show()

