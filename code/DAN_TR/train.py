#-*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from BG_loader import BG_Dataset_Labeled
from torch.utils.data import DataLoader
import DAN
import numpy as np
import os, math
import time, datetime
import mmd
import random as rd

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch DAAN')

args = parser.parse_args()
args.gpu = "0" # gpu device to be used
args.series = "file_name" # file name to be saved

DEVICE = torch.device('cuda:' + str(args.gpu))
args.device = DEVICE

args.batch_size = 200 # batch_size for dataset
args.epochs = 200 # number of maximum epoches
args.LR = 5e-3 # learning rate

args.root_path_s = "G:\清华云盘\TL\伪标签数据" # dataset path for source domain, target domain, and validation dataset
args.root_path_t = "G:\清华云盘\TL\伪标签数据"
args.root_path_t_test = args.root_path_t

args.mat_file_s = "un_data_S10_inte_1.mat" # unlabeled dataset
args.mat_file_t = "data_Huang_norm_final_10.mat" # labeled dataset
args.mat_file_t_test = args.mat_file_t # validation dataset

args.key_X_s, args.key_X_t = "X_train", "X_train" # unlabeled dataset keys
args.key_Y_s, args.key_Y_t = "y_train", "y_train" # labeled dataset keys
args.key_X_t_test, args.key_Y_t_test = "X_valid", "y_valid" # validation dataset keys


def load_data(): # load the dataset of unlabeled, labeled and validation datasets
    source_train_loader = BG_Dataset_Labeled(args.root_path_s, args.mat_file_s, args.key_X_s, args.key_Y_s)
    target_train_loader = BG_Dataset_Labeled(args.root_path_t, args.mat_file_t, args.key_X_t, args.key_Y_t)
    target_valid_loader = BG_Dataset_Labeled(args.root_path_t_test, args.mat_file_t_test, args.key_X_t_test, args.key_Y_t_test)
    print("source train data:", source_train_loader.len, "\ntarget train data:", target_train_loader.len,
          "\ntarget valid data:",target_valid_loader.len)
    Dataset_s = DataLoader(source_train_loader, batch_size=args.batch_size, shuffle=True, num_workers=0)
    Dataset_t = DataLoader(target_train_loader, batch_size=args.batch_size, shuffle=True, num_workers=0)
    Dataset_t_v = DataLoader(target_valid_loader, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return Dataset_s, Dataset_t, Dataset_t_v


def train(epoch, model, optimizer, source_loader, target_loader):

    model.train()
    len_dataloader = len(source_loader)
    s_loss, t_loss, d_loss = 0, 0, 0
    step = 0
    iteration = min(args.epochs, 1000) * len_dataloader

    for batch_idx, (source_data, source_label) in enumerate(source_loader):

        source_data, source_label = source_data.to(DEVICE).float(), source_label.to(DEVICE).float()

        for target_data, target_label in target_loader:
            target_data, target_label = target_data.to(DEVICE).float(), target_label.to(DEVICE).float()
            break

        batch_size_pre = source_data.size(0)

        mmd_st, st_output, s_output, mmd_t, t_output = model(source_data, target_data)
        # forward, the outputs in sequences are: hidden state for mmd calculation in st branch, output in st branch, output in s branch, hidden state for mmd calculation in t branch, output in t branch
        # t branch is the target domain branch, s branch is the source domain branch, st in the transfer branch
        des_loss_batch = 0.0

        for id_mmd in range(len(mmd_st)):

            des_loss = mmd.mmd_rbf_noaccelerate(mmd_st[id_mmd], mmd_t[id_mmd]) # MMD Loss, see mmd.py
            des_loss_batch += des_loss

        i = min(epoch * len_dataloader + batch_idx, iteration)
        lambd = min(2 / (1 + math.exp(-10 * (i) / iteration)) - 1, 3) # weight of the MMD Loss

        s_loss_batch = (F.mse_loss(s_output, source_label) * source_data.size(0)
                        + F.mse_loss(st_output, target_label) * target_data.size(0)) / (source_data.size(0) + target_data.size(0))
        t_loss_batch = F.mse_loss(t_output, target_label)

        loss_batch = s_loss_batch + t_loss_batch + lambd * 1 * des_loss_batch

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        s_loss += s_loss_batch.item() * batch_size_pre
        t_loss += t_loss_batch.item() * batch_size_pre
        d_loss += des_loss_batch.item() * batch_size_pre
        step += batch_size_pre

    s_loss /= step
    t_loss /= step
    d_loss /= step

    return s_loss, t_loss, d_loss


def check_part(model, test_loader): # validation function
    model.eval()
    test_loss_1, test_loss_2 = 0, 0

    step_test = 0
    with torch.no_grad():
        for test_data, test_label in test_loader:
            test_data, test_label = test_data.to(DEVICE).float(), test_label.to(DEVICE).float()

            _, _, s_output, _, t_output = model(test_data, test_data)
            batch_loss_1 = F.mse_loss(s_output, test_label)
            batch_loss_2 = F.mse_loss(t_output, test_label)
            test_loss_1 += batch_loss_1.item() * test_data.size(0)
            test_loss_2 += batch_loss_2.item() * test_data.size(0)

            step_test += test_data.size(0)

    test_loss_1 /= step_test
    test_loss_2 /= step_test
    return test_loss_1, test_loss_2


def write_log_file(epoch, loss_list, series, time):
    f = open("./log/log_%s.csv" % series, "a")

    f.write("%s,%s," %(epoch, time))
    for loss_item in loss_list:
        f.write("%s," % loss_item)
    f.write("\n")
    f.close()


def print_performance(epoch, type, loss):
    print("\nEpoch: ", epoch)
    assert len(type) == len(loss)
    for (loss_type, loss_value) in zip(type, loss):
        print("%s: %.8f" % (loss_type, loss_value * 1000))


if __name__ == '__main__':

    model = DAN.DANNet(args).to(DEVICE)
    # print(model)
    train_s, train_t, valid_t = load_data()

    print(args.mat_file_s, "to", args.mat_file_t)
    test_loss_min = 1e9
    count = 0

    optimizer = optim.Adam(model.parameters(), lr=args.LR, betas=(0.9, 0.99), eps=1e-09)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    if not os.path.exists("./model"):
        os.mkdir("./model")
    if not os.path.exists("./log"):
        os.mkdir("./log")

    if not os.path.exists("./log/log_%s.csv" % args.series):
        f = open("./log/log_%s.csv" % args.series, "w")
        f.close()
    else:
        raise AssertionError


    for epoch in range(1, args.epochs + 1):
        s_loss, t_loss, d_loss = train(epoch, model, optimizer, train_s, train_t)
        test_loss_1, test_loss_2 = check_part(model, valid_t)
        scheduler.step()
        loss_type = "s_loss, t_loss, d_loss, test_loss_1, test_loss_2".split(", ")
        loss_list = [s_loss, t_loss, d_loss, test_loss_1, test_loss_2]
        print_performance(epoch, loss_type, loss_list)

        timestamp = time.time()
        dateArray = datetime.datetime.fromtimestamp(timestamp)
        Styletime = dateArray.strftime("%Y-%m-%d %H:%M:%S")

        write_log_file(epoch, loss_list, args.series, Styletime)
        if min(test_loss_1, test_loss_2) < test_loss_min:
            test_loss_min = min(test_loss_1, test_loss_2)

            state = {"net": model.state_dict(), "epoch": epoch}
            torch.save(state, "./model/model_best_%s.chkpt" % (args.series))

            print("model updated at %s" % Styletime)
            count += 1