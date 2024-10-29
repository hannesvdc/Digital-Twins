import torch as pt
import torch.fft as fft

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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

def normalize(network, y, y_x, y_xx, y_xxx, y_xxxx):
    return (y - network.y_mean) / network.y_std, \
           (y_x - network.y_x_mean) / network.y_x_std, \
           (y_xx - network.y_xx_mean) / network.y_xx_std, \
           (y_xxx - network.y_xxx_mean) / network.y_xxx_std, \
           (y_xxxx - network.y_xxxx_mean) / network.y_xxxx_std

def unnormalize(network, yt):
    return network.y_t_mean + yt * network.y_t_std

def rhs(network, y, k):
    y_x, y_xx, y_xxx, y_xxxx = calcSpatialDerivatives(y, k)
    y_x, y_xx, y_xxx, y_xxxx = normalize(network, y, y_x, y_xx, y_xxx, y_xxxx)
    input_data = pt.vstack((y, y_x, y_xx, y_xxx, y_xxxx)).transpose(0, 1)

    output_data = network.forward(input_data)[:,0]
    return unnormalize(network, output_data)

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
dt = 1.e-5
T = 100.0
N = int(T / dt)
store_n = 1000
for n in range(N):
    y = RK4(network, y, k, dt)

    if n % store_n == 0: # Store every 0.01 seconds
        print('t =', (n+1)*dt)
        time_simulation.append(y)

# Make a movie
x_array = np.linspace(0.0, L, M)
image_folder = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/WaveData/_tmp_images/'
for n in range(len(time_simulation)):
    fig = plt.figure()
    plt.title(r'$T = $' + str(n * store_n * dt))
    plt.plot(x_array, time_simulation[n])
    plt.savefig(image_folder + str(n) + '_img.png')
    (width, height) = fig.canvas.get_width_height()
    plt.close()

video_folder = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/WaveData/'
video_name = 'KS_BlackBox.avi'
fps = 100
video = cv2.VideoWriter(video_folder + video_name, 0, fps, (width, height))
for n in range(len(time_simulation)):
    video.write(cv2.imread(os.path.join(image_folder, str(n) + '_img.png')))
cv2.destroyAllWindows()
video.release()
