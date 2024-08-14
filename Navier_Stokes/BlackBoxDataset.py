import torch as pt
import numpy as np
import numpy.fft as fft

from torch.utils.data import Dataset

class NSDataSet(Dataset):
    def __init__(self):
        super(NSDataSet, self).__init__()

        # Load the dataset for this specific Reynolds Number
        self.R_string = '1p75'
        self.R = 1.75
        self.storage_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/WaveData/'
        self.y_filename = 'newRe' + self.R_string + '_y.dat'
        self.dydt_filename = 'newRe' + self.R_string + '_dydt.dat'
        y_data = np.loadtxt(self.storage_directory + self.y_filename)
        dydt_data = np.loadtxt(self.storage_directory + self.dydt_filename)
        self.N_data = y_data.shape[0]

        # Compute the spatial Derivatives
        self.L = 95.0
        self.M = y_data.shape[1] # 1001
        self.k = np.concatenate((np.arange(self.M // 2 + 1), np.arange(-self.M // 2 + 1, 0))) * 2.0 * np.pi / self.L
        dydx_data = np.zeros_like(y_data)
        dydxx_data = np.zeros_like(y_data)
        dydxxx_data = np.zeros_like(y_data)
        dydxxxx_data = np.zeros_like(y_data)
        for index in range(y_data.shape[0]):
            f_eta = fft.fft(y_data[index,:])
            f_eta_x = 1j * self.k * f_eta
            f_eta_xx = (1j * self.k)**2 * f_eta
            f_eta_xxx = (1j * self.k)**3 * f_eta
            f_eta_xxxx = (1j * self.k)**4 * f_eta
            dydx_data[index, :] = np.real(fft.ifft(f_eta_x))
            dydxx_data[index, :] = np.real(fft.ifft(f_eta_xx))
            dydxxx_data[index, :] = np.real(fft.ifft(f_eta_xxx))
            dydxxxx_data[index, :] = np.real(fft.ifft(f_eta_xxxx))

        # Convert spatial derivatives to pytorch without gradients
        pt.set_grad_enabled(False)
        self.output_data = pt.unsqueeze(pt.from_numpy(dydt_data.flatten()), dim=1)
        self.input_data = pt.zeros((y_data.size, 5))
        self.input_data[:,0] = pt.from_numpy(y_data.flatten())
        self.input_data[:,1] = pt.from_numpy(dydx_data.flatten())
        self.input_data[:,2] = pt.from_numpy(dydxx_data.flatten())
        self.input_data[:,3] = pt.from_numpy(dydxxx_data.flatten())
        self.input_data[:,4] = pt.from_numpy(dydxxxx_data.flatten())
        pt.set_grad_enabled(True)

    def __len__(self):
        return self.N_data
    
    def __getitem__(self, idx):
        return (self.input_data[idx, :], self.output_data[idx, :])

if __name__ == '__main__':
    dataset = NSDataSet()
