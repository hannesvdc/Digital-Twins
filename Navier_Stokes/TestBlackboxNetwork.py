import json
import torch as pt
import torch.fft as fft

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from BlackBoxModel import FeedforwardNetwork
from BlackBoxDataset import NSDataSet

# Just some sanity pytorch settings
pt.set_grad_enabled(False)
device = pt.device("cpu")
dtype = pt.float32

# Load the global dataset
dataset = NSDataSet(device, dtype)

def calcSpatialDerivatives(y, k):
    f = fft.fft(y)
    y_x = pt.real(fft.ifft( 1j * k * f ))
    y_xx = pt.real(fft.ifft( (1j * k)**2 * f ))
    y_xxx = pt.real(fft.ifft( (1j * k)**3 * f ))
    y_xxxx = pt.real(fft.ifft( (1j * k)**4 * f ))
    
    return y_x, y_xx, y_xxx, y_xxxx

def rhs(network, y, k, R):
    # Parse the input
    y_x, y_xx, y_xxx, y_xxxx = calcSpatialDerivatives(y, k)
    nn_input = pt.stack([y, y_x, y_xx, y_xxx, y_xxxx, R * pt.ones_like(y)], dim=1)
    nn_input = (nn_input - dataset.input_mean) / dataset.input_std

    nn_output = network.forward(nn_input)
    nn_output = dataset.output_mean + nn_output * dataset.output_std
    return nn_output[:,0]

def RK4(network, y, k, R, dt):
    k1 = rhs(network, y, k, R)
    k2 = rhs(network, y + 0.5*dt*k1, k, R)
    k3 = rhs(network, y + 0.5*dt*k2, k, R)
    k4 = rhs(network, y + 1.0*dt*k3, k, R)

    return y + dt * (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

# Load the initial condition
print('Loading Initial Condition.')
config = json.load(open('DataConfig.json', 'r'))
data_directory = config['Data Directory']
results_directory = config['Results Directory']
y_filename = 'newRe1p75_y.dat'
y0 = pt.tensor(np.loadtxt(data_directory + y_filename)[:,0], dtype=dtype, device=device)

# Model parameters
R = 1.75
L = 95.0
M = list(y0.size())[0]
k = pt.tensor(np.concatenate((np.arange(M // 2 + 1), np.arange(-M // 2 + 1, 0))) * 2.0 * np.pi / L, dtype=dtype, device=device)

# Load the optimized network state
network = FeedforwardNetwork()
network.load_state_dict(pt.load(results_directory + 'model_black_box.pth', weights_only=True))

# Simulate the KS equations and store the intermediate solutions
y = pt.clone(y0)
dt = 1.e-5
T = 100.0
n_steps = int(T / dt)
store_n = 1000
time_simulation = np.zeros((n_steps // store_n + 1, M))
time_simulation[0,:] = np.copy(y.cpu().numpy())
for n in range(1, n_steps+1):
    y = RK4(network, y, k, R, dt)

    if n % store_n == 0: # Store every 0.01 seconds
        print('t =', (n+1)*dt)
        store_index = n // store_n
        time_simulation[store_index, :] = np.copy(y.cpu().numpy())

# Make a movie
x_array = np.linspace(0.0, L, M)
image_folder = data_directory + '/_tmp_images/'
for n in range(len(time_simulation)):
    fig = plt.figure()
    plt.title(r'$T = $' + str(round(n * store_n * dt, 2)))
    plt.plot(x_array, time_simulation[n,:])
    plt.savefig(image_folder + str(n) + '_img.png')
    (width, height) = fig.canvas.get_width_height()
    plt.close()

video_name = 'KS_BlackBox.avi'
fps = 100
video = cv2.VideoWriter(data_directory + video_name, 0, fps, (width, height))
for n in range(len(time_simulation)):
    video.write(cv2.imread(os.path.join(image_folder, str(n) + '_img.png')))
cv2.destroyAllWindows()
video.release()
