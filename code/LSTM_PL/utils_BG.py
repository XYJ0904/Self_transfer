import random as rd
import numpy as np
from BG_loader import BG_Dataset_Labeled
from torch.utils.data import DataLoader
import torch
import os
from torch.autograd import Variable

def prepare_dataloaders(opt):

    training_data, validation_data, train_len, valid_len = prepare_labled_dataloaders(opt)
    # unlabeled_data, unlabeled_len = prepare_unlabeled_dataloaders(opt)
    print("training dataset size: %s samples" % train_len)
    print("validation dataset size: %s samples" % valid_len)

    return training_data, validation_data, None


def prepare_labled_dataloaders(opt):
    batch_size = opt.batch_size

    train_data = BG_Dataset_Labeled(opt.root_path_T, opt.mat_file_T, opt.key_X_train, opt.key_y_train)
    validate_data = BG_Dataset_Labeled(opt.root_path_V, opt.mat_file_V, opt.key_X_valid, opt.key_y_valid)

    train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_iterator = DataLoader(validate_data, batch_size=40, shuffle=False, num_workers=0)

    return train_iterator, val_iterator, train_data.len, validate_data.len


def cal_loss(pred, real_value):
    # Calculate cross entropy loss, apply label smoothing if needed.
    loss_func = torch.nn.MSELoss(reduction="mean")
    loss = loss_func(pred, real_value)

    return loss