''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np


class LSTM_model(nn.Module):

    def __init__(self):
        super().__init__()
        # self.LSTM = nn.Sequential()
        power = 2
        self.dim = 200
        self.dim_half = self.dim / power
        self.input = 1
        self.output = 1

        self.LSTM_1 = nn.LSTM(input_size=self.input, hidden_size=self.dim, num_layers=2, batch_first=True)
        self.activeation_1 = nn.ReLU()
        self.LSTM_2 = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=1, batch_first=True)
        self.activeation_2 = nn.ReLU()

        self.Linear_3 = nn.Linear(self.dim, self.dim)
        self.Linear_4 = nn.Linear(self.dim, self.output)
        self.activeation_3 = nn.ReLU()

    def forward(self, x):

        self.LSTM_1.flatten_parameters()
        self.LSTM_2.flatten_parameters()

        LSTM_output_1, _ = self.LSTM_1(x)
        LSTM_output_1 = self.activeation_1(LSTM_output_1)
        LSTM_output_2, _ = self.LSTM_2(LSTM_output_1)
        LSTM_output_2 = self.activeation_2(LSTM_output_2)

        # see : https://doi.org/10.48550/arXiv.2206.03990
        LSTM_output = (LSTM_output_1 * self.dim_half + LSTM_output_2 * self.dim) / (self.dim + self.dim_half)

        LSTM_output = self.Linear_3(LSTM_output)
        LSTM_output = self.activeation_3(LSTM_output)
        output = self.Linear_4(LSTM_output)

        return output