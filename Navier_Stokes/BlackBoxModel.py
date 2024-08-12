import torch as pt
import numpy as np
import numpy.fft as fft

from torch.utils.data import Dataset, DataLoader

class NSDataSet(Dataset):
    def __init__(self):
        super(NSDataSet, self).__init__()

        # Load the dataset for this specific Reynolds Number
        self.R_string = '1p75'
        self.R = 1.75
        self.storage_directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/WaveData/'
        self.y_filename = 'newRe' + self.R_string + '_y.dat'
        self.dydt_filename = 'newRe' + self.R_string + '_dydt.dat'
        self.y_data = np.loadtxt(self.storage_directory + self.y_filename)
        self.dydt_data = np.loadtxt(self.storage_directory + self.dydt_filename)
        self.N_data = self.y_data.shape[0]

        # Compute the spatial Derivatives
        self.L = 95.0
        self.M = self.y_data.shape[1] # 1001
        self.k = np.concatenate((np.arange(self.M // 2 + 1), np.arange(-self.M // 2 + 1, 0))) * 2.0 * np.pi / self.L
        self.dydx_data = np.zeros_like(self.y_data)
        self.dydxx_data = np.zeros_like(self.y_data)
        self.dydxxx_data = np.zeros_like(self.y_data)
        self.dydxxxx_data = np.zeros_like(self.y_data)
        for index in range(self.y_data.shape[0]):
            f_eta = fft.fft(self.y_data[index,:])
            f_eta_x = 1j * self.k * f_eta
            f_eta_xx = (1j * self.k)**2 * f_eta
            f_eta_xxx = (1j * self.k)**3 * f_eta
            f_eta_xxxx = (1j * self.k)**4 * f_eta
            self.dydx_data[index, :] = np.real(fft.ifft(f_eta_x))
            self.dydxx_data[index, :] = np.real(fft.ifft(f_eta_xx))
            self.dydxxx_data[index, :] = np.real(fft.ifft(f_eta_xxx))
            self.dydxxxx_data[index, :] = np.real(fft.ifft(f_eta_xxxx))

        # Convert spatial derivatives to pytorch without gradients
        has_grad = pt.is_grad_enabled()
        pt.set_grad_enabled(False)
        self.dydt_data = pt.from_numpy(self.dydt_data)
        self.y_data = pt.from_numpy(self.y_data)
        self.dydx_data = pt.from_numpy(self.dydx_data)
        self.dydxx_data = pt.from_numpy(self.dydxx_data)
        self.dydxxx_data = pt.from_numpy(self.dydxxx_data)
        self.dydxxxx_data = pt.from_numpy(self.dydxxxx_data)
        pt.set_grad_enabled(has_grad)

    def __len__(self):
        return self.N_data
    
    def __getitem__(self, idx):
        return (self.y_data[idx,:], self.dydx_data[idx,:], self.dydxx_data[idx,:], self.dydxxx_data[idx,:], self.dydxxxx_data[idx,:]), self.dydt_data[idx,:]

if __name__ == '__main__':
    dataset = NSDataSet()
