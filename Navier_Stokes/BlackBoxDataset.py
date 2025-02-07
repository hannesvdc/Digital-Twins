import torch as pt
import numpy as np
import numpy.fft as fft
import json

from torch.utils.data import Dataset

class NSDataSet(Dataset):
    def __init__(self, device, dtype):
        super(NSDataSet, self).__init__()

        # Geometry Parameters
        L = 95.0
    
        # Load the Data Config file
        dataConfigFile = open("DataConfig.json")
        dataConfig = json.load(dataConfigFile)
        storage_directory = dataConfig["Data Directory"]
        print('Data directory', storage_directory)

        # Load the dataset for this specific Reynolds Number. Calculate the data size first.
        self.data_size = 0
        R_values = ['1p15', '1p35', '1p75', '1p85', '1p95', '2p05', '2p15', '2p25', '2p45', '2p60', '3p0', '3p2', '3p5', '3p7', '4p0', '4p2', '4p5', '4p7', '4p9', '5p2']
        for index in range(len(R_values)):
            R_string = R_values[index]
            y_filename = 'newRe' + R_string + '_y.dat'
            dydt_filename = 'newRe' + R_string + '_dydt.dat'
            y_data = np.loadtxt(storage_directory + y_filename).flatten('F')
            dydt_data = np.loadtxt(storage_directory + dydt_filename).flatten('F')
            self.data_size += y_data.size
        print('Total Data Size:', self.data_size)

        # Then load the data
        data_index = 0
        self.input_data = pt.zeros((self.data_size, 6), dtype=dtype, device=device, requires_grad=False)
        self.output_data = pt.zeros((self.data_size, 1), dtype=dtype, device=device, requires_grad=False)
        for index in range(len(R_values)):
            R_string = R_values[index]
            R = float(R_string.replace('p', '.'))
            y_filename = 'newRe' + R_string + '_y.dat'
            dydt_filename = 'newRe' + R_string + '_dydt.dat'
            y_data = np.loadtxt(storage_directory + y_filename)
            dydt_data = np.loadtxt(storage_directory + dydt_filename)

            # Compute the spatial Derivatives
            M = y_data.shape[0]
            k = np.concatenate((np.arange(M // 2 + 1), np.arange(-M // 2 + 1, 0))) * 2.0 * np.pi / L
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

            # Convert spatial derivatives to pytorch without gradients
            block_data_size = y_data.size
            input_block_data = pt.zeros((block_data_size, 6), dtype=dtype, device=device)
            input_block_data[:,0] = pt.tensor(y_data, device=device, dtype=dtype)
            input_block_data[:,1] = pt.tensor(dydx_data, device=device, dtype=dtype)
            input_block_data[:,2] = pt.tensor(dydxx_data, device=device, dtype=dtype)
            input_block_data[:,3] = pt.tensor(dydxxx_data, device=device, dtype=dtype)
            input_block_data[:,4] = pt.tensor(dydxxxx_data, device=device, dtype=dtype)
            input_block_data[:,5] = R * pt.ones(block_data_size, device=device, dtype=dtype)
            output_block_data = pt.unsqueeze(pt.tensor(dydt_data, device=device, dtype=dtype), dim=1)

            # Store in the big data tensor
            self.input_data[data_index:data_index+block_data_size,:] = input_block_data
            self.output_data[data_index:data_index+block_data_size,:] = output_block_data
            data_index += block_data_size

        # Normalize the data for better training
        self.input_mean = pt.mean(self.input_data, dim=0, keepdim=True)
        self.input_std = pt.std(self.input_data, dim=0, keepdim=True)
        self.output_mean = pt.mean(self.output_data, dim=0, keepdim=True)
        self.output_std = pt.std(self.output_data, dim=0, keepdim=True)
        self.input_data = (self.input_data - self.input_mean) / self.input_std
        self.output_data = (self.output_data - self.output_mean) / self.output_std

        # Some bookkeeping statistics
        print('Total Storage Size:', (self.input_data.nelement() * self.input_data.element_size() + self.output_data.nelement() * self.output_data.element_size()) / 1024.0**2, 'MB')

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        return (self.input_data[idx, :], self.output_data[idx, :])

if __name__ == '__main__':
    dataset = NSDataSet('cpu', pt.float64)
