import argparse
import time
import numpy as np
import random as rd
import os
import torch
import torch.nn.functional as F
import copy
import scipy.io as io
import DAN as DAN
from BG_loader import BG_Dataset_Labeled
from torch.utils.data import DataLoader


# this program is designed to assign the program to unlabeled samples and establish the training dataset for next iteration

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def generate_para():
    global global_step
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    opt.root_path_s = "G:\清华云盘\TL\伪标签数据"

    # opt.mat_file_s = "unlabeled.mat" # unlabeled dataset
    opt.mat_file_s = "data_Huang_norm_final_10.mat"  # unlabeled dataset
    opt.key_X_s = "X_train"

    opt.cuda = True
    opt.batch_size = 100
    opt.random_seed = "1"

    if not os.path.exists("./pred_result_un"):
        os.mkdir("./pred_result_un")

    return opt


def cal_loss(pred, real_value):
    # Calculate cross entropy loss, apply label smoothing if needed.
    loss_func = torch.nn.MSELoss(reduction="mean")
    loss = loss_func(pred, real_value)

    return loss


def load_data(args): # 加载数据
    source_train_loader = BG_Dataset_Labeled(args.root_path_s, args.mat_file_s, args.key_X_s, args.key_X_s) # unlabeled dataset do not have "y_train"
    Dataset_s = DataLoader(source_train_loader, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return Dataset_s


def eval_epoch(model, input_data, type, opt, model_type):
    model.eval()

    def_all_train = []
    force_all_train = []

    with torch.no_grad():
        for src_seq_v, _ in input_data: # no "target" for unlabeled data

            src_seq_v = src_seq_v.cuda().float()
            t_output = model(src_seq_v, src_seq_v)

            if model_type == "s": # the output type depends the branch (source branch or target branch) adopted in assigning pesudo label, see DAN.py
                pred_seq_v = t_output[2]
            elif model_type == "t":
                pred_seq_v = t_output[-1]

            def_all_train.append(src_seq_v.detach().cpu().numpy())
            force_all_train.append(pred_seq_v.detach().cpu().numpy())

    def_all_train = np.concatenate(def_all_train, axis=0)
    force_all_train = np.concatenate(force_all_train, axis=0)

    print(def_all_train.shape, force_all_train.shape)
    io.savemat("./pred_result_un/un_data_%s_%s.mat" % (opt.folder, type),
                     {'y_train': force_all_train, 'X_train': def_all_train})


def main():

    opt = generate_para()
    all_models = ["model_test.chkpt"]
    model_type = "t" # a flag which is added in the name of the prediction results
    train_s = load_data(opt)
    model = DAN.DANNet(opt).cuda() # if this model is replaced by the LSTM model, then it could be used for pesudo-labeled dataset generation based on the LSTM model

    for file in all_models:
        print("Model name:", file, "\n", "dataset name:", opt.mat_file_s)
        opt.checkpoint_load = file
        opt.folder = opt.checkpoint_load.rstrip("chkpt").rstrip(".")
        checkpoint = torch.load("./model/%s" % opt.checkpoint_load, map_location="cuda:0")
        model.load_state_dict(checkpoint['net'])

        eval_epoch(model, train_s, "un_10_L", opt, model_type)


if __name__ == '__main__':
    main()
