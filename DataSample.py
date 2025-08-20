import os
from tkinter import Y
# from scipy.io import loadmat, savemat
import pandas as pd
import scipy.io as scio
import torch
import numpy as np
from torch.utils.data import Dataset


class Input_Data(Dataset):
    def __init__(self, params, recording_data, stim_data, hidden_data=None):
        # initialize with input data, Channels*T
        super().__init__()

        self.Cortical = params['Num_Cortical']  # number of recording channels
        self.tp_step = params['Tp']
        self.data_length = params['data_length']  # The data length for parameter estimation

        self.ext_input_dim = params['ext_input_dim'] # number of stimulating channels
        # print(self.Data_Path)
        data = np.array(recording_data)[:self.Cortical, :]  # Cortical*T
        self.hidden_data = np.transpose(hidden_data)
        self.data = np.transpose(data)
        self.max_value = max(self.data_length, self.tp_step)

        self.datalen = self.data.shape[0] - self.max_value - 1

        self.X = self.data[:-1,:]
        self.ext_input = np.transpose(np.array(stim_data))[1:, :]

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        """
        :X     : Batch x (Length+Tp) x Dim
        :Ext_In: Batch x (Length+Tp) x Ext_Dim
        :return:
        """
        index = int(index)
        X = torch.from_numpy(
            self.X[index:(index + self.data_length), :]).to(torch.float32)
        ext_in = torch.from_numpy(
            self.ext_input[index:(index + self.data_length),:]).to(torch.float32)
        # print(X.shape,ext_in.shape)
        return X, ext_in, index

