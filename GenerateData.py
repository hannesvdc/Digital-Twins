import numpy as np
import scipy.linalg as lg

from ElasticPDE import *
from GaussianProces import *

# Model parameters
N = 100           # N is the number of intervals
E = 410.0 * 1.e3 # This is really, really large?
mu = 0.3
dx = 1.0 / N

# Load matrix into memory
directory = '/Users/hannesvdc/Research_Data/Digital Twins/Elastodynamics/'
filename = 'FD_Matrix_mu=' + str(mu) + '.npy'
try:
    print('Attempting Loading System Matrix into Memory...')
    A = np.load(directory + filename)
    print('Loading Succesful!')
except:
    print('Computing and Storing the Finite Differences System Matrix.')
    A = computeSystemMatrix(mu, N=N)
    np.save(directory + filename, A)

# Compute LU factorization of A
print('\nComputing LU Factorization...')
lu, pivot = lg.lu_factor(A)
print('Done.')

# Generate 100 random (x,y) points
N_trunk_points = (N + 1) * (N + 1)
trunk_data = np.zeros((2, N_trunk_points))
counter = 0
for i in range(N+1):
    for j in range(N+1):
        x = i * dx
        y = j * dx
        trunk_data[:,counter] = np.array([x, y])

        counter += 1
print('\nNumber of (x,y) points:', trunk_data.shape[1])

# Setup the Gaussian Process to generate branch data
l = 0.12
y_points = np.linspace(0.0, 1.0, N+1)
K = precompute_covariance(y_points, l)

# Generate right-hand side data and
N_branch_samples = 100
branch_data = np.zeros((202, N_branch_samples))
G = np.zeros((N_branch_samples, N_trunk_points))
for i in range(N_branch_samples):
    f1, f2 = gp(y_points, K)
    branch_data[:,i] = np.concatenate((f1, f2))

    f = np.column_stack((f1, f2))
    (u, v) = solveElasticPDE(lu, pivot, E, mu, f, N)
    print('x=0', u[:,0])

    for j in range(N_trunk_points):
        x = int(trunk_data[0,j] / dx) # x, y are integer indices between 0 and N
        y = int(trunk_data[1,j] / dx)
        G[i, j] = u[y,x] # u contains zeros on the first column

print('Shape of G', G.shape, branch_data.shape)
print(G)

# Store all training data in separate files
directory = '/Users/hannesvdc/Research_Data/Digital Twins/Elastodynamics/'
f_filename = 'branch_data.npy'
y_filename = 'trunk_data.npy'
g_filename = 'output_data.npy'
np.save(directory + f_filename, branch_data)
np.save(directory + y_filename, trunk_data)
np.save(directory + g_filename, G)

