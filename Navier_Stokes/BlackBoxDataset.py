import torch as pt
import numpy as np
import numpy.fft as fft
import scipy.stats as stats

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

        # Compute the spatial Derivatives
        self.L = 95.0
        self.M = y_data.shape[0]
        k = np.concatenate((np.arange(self.M // 2 + 1), np.arange(-self.M // 2 + 1, 0))) * 2.0 * np.pi / self.L
        dydx_data = np.zeros_like(y_data)
        dydxx_data = np.zeros_like(y_data)
        dydxxx_data = np.zeros_like(y_data)
        dydxxxx_data = np.zeros_like(y_data)
        for index in range(y_data.shape[1]):
            f_eta = fft.fft(y_data[:, index])
            f_eta_x    = (1j * k) * f_eta
            f_eta_xx   = (1j * k)**2 * f_eta
            f_eta_xxx  = (1j * k)**3 * f_eta
            f_eta_xxxx = (1j * k)**4 * f_eta
            dydx_data[:, index] = np.real(fft.ifft(f_eta_x))
            dydxx_data[:, index] = np.real(fft.ifft(f_eta_xx))
            dydxxx_data[:, index] = np.real(fft.ifft(f_eta_xxx))
            dydxxxx_data[:, index] = np.real(fft.ifft(f_eta_xxxx))
        y_data = y_data.flatten('F')
        dydt_data = dydt_data.flatten('F')
        dydx_data = dydx_data.flatten('F')
        dydxx_data = dydxx_data.flatten('F')
        dydxxx_data = dydxxx_data.flatten('F')
        dydxxxx_data = dydxxxx_data.flatten('F')

        self.y_mean, self.y_std = np.mean(y_data), np.std(y_data)
        self.y_t_mean, self.y_t_std = np.mean(dydt_data), np.std(dydt_data)
        self.y_x_mean, self.y_x_std = np.mean(dydx_data), np.std(dydx_data)
        self.y_xx_mean, self.y_xx_std = np.mean(dydxx_data), np.std(dydxx_data)
        self.y_xxx_mean, self.y_xxx_std = np.mean(dydxxx_data), np.std(dydxxx_data)
        self.y_xxxx_mean, self.y_xxxx_std = np.mean(dydxxxx_data), np.std(dydxxxx_data)

        # Convert spatial derivatives to pytorch without gradients
        self.input_data = pt.zeros((y_data.size, 5)).requires_grad_(False)
        self.input_data[:,0] = pt.from_numpy((y_data - self.y_mean) / self.y_std)
        self.input_data[:,1] = pt.from_numpy((dydx_data - self.y_x_mean) / self.y_x_std)
        self.input_data[:,2] = pt.from_numpy((dydxx_data - self.y_xx_mean) / self.y_xx_std)
        self.input_data[:,3] = pt.from_numpy((dydxxx_data - self.y_xxx_mean) / self.y_xxx_std)
        self.input_data[:,4] = pt.from_numpy((dydxxxx_data - self.y_xxxx_mean) / self.y_xxxx_std)
        self.output_data = pt.unsqueeze(pt.from_numpy( (dydt_data - self.y_t_mean) / self.y_t_std), dim=1).requires_grad_(False)

    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        return (self.input_data[idx, :], self.output_data[idx, :])

if __name__ == '__main__':
    dataset = NSDataSet()
