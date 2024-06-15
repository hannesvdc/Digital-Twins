import jax.numpy as np

from api.DenseNN import DenseNN

class DeepONet:
    def __init__(self, branch_layers, trunk_layers, activation):
        self.branch_net = DenseNN(branch_layers, activation)
        self.trunk_net  = DenseNN(trunk_layers, activation)
        self.n_branch_weights = self.branch_net.n_weights
        self.n_trunk_weights = self.trunk_net.n_weights
        self.n_weights = self.n_branch_weights + self.n_trunk_weights

    def forward(self, branch_data, trunk_data, weights):
        branch_p = self.branch_net.forward(branch_data, weights[:self.n_branch_weights]) # Shape (p, N_branch_data)
        trunk_p  = self.trunk_net.forward(trunk_data, weights[self.n_branch_weights:])  # Shape (p, N_trunk_data)

        # Broadcast both matrices to same size
        N_branch_data = branch_data.shape[1]
        N_trunk_data = trunk_data.shape[1]
        branch_p = np.repeat(branch_p[:,:,np.newaxis], N_trunk_data, axis=2)
        trunk_p  = np.repeat(trunk_p[:,np.newaxis,:], N_branch_data, axis=1)

        # Multiply element-wise and return the sum over all p
        return np.sum(np.multiply(branch_p, trunk_p), axis=0)