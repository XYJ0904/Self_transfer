from glob import glob
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random as rd
from scipy.io import loadmat


class BG_Dataset_Labeled(Dataset):
    def __init__(self, root_path, mat_file, key_X, key_Y):
        raw_data = loadmat("%s/%s" % (root_path, mat_file))
        X = raw_data[key_X]
        Y = raw_data[key_Y]

        self.len = X.shape[0]
        assert self.len == Y.shape[0]

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


if __name__ == '__main__':
    pass
