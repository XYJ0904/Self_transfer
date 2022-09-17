import argparse
import time
import numpy as np
import random as rd
import os
import torch
import torch.nn.functional as F
import copy
import scipy.io as io
from BG_loader import BG_Dataset_Labeled
from torch.utils.data import DataLoader
import DAN as DAN


# this program is used for validation and testing, which helps the users to verify the model performance

global_step = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def generate_para():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()

    opt.root_path_s = "G:\清华云盘\TL\伪标签数据" # path
    opt.root_path_t = "G:\清华云盘\TL\伪标签数据"
    opt.root_path_t_test = opt.root_path_t

    opt.mat_file_s = "un_data_S10_inte_2.mat" # file name
    opt.mat_file_t = "data_Huang_norm_final_10.mat"
    opt.mat_file_t_test = opt.mat_file_t

    opt.key_X_s, opt.key_Y_s = "X_train", "y_train" # keys in the .mat file
    opt.key_X_t, opt.key_Y_t = "X_valid", "y_valid"
    opt.key_X_t_test, opt.key_Y_t_test = "X_test", "y_test"

    opt.cuda = True
    opt.batch_size = 100
    opt.random_seed = "1"

    if not os.path.exists("./pred_result"):
        os.mkdir("./pred_result")

    return opt


def load_data(args): # 加载数据
    source_train_loader = BG_Dataset_Labeled(args.root_path_s, args.mat_file_s, args.key_X_s, args.key_Y_s)
    target_train_loader = BG_Dataset_Labeled(args.root_path_t, args.mat_file_t, args.key_X_t, args.key_Y_t)
    target_valid_loader = BG_Dataset_Labeled(args.root_path_t_test, args.mat_file_t_test, args.key_X_t_test, args.key_Y_t_test)
    Dataset_s = DataLoader(source_train_loader, batch_size=args.batch_size, shuffle=False, num_workers=0)
    Dataset_t = DataLoader(target_train_loader, batch_size=args.batch_size, shuffle=False, num_workers=0)
    Dataset_t_v = DataLoader(target_valid_loader, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return Dataset_s, Dataset_t, Dataset_t_v


 # as long as both the source domain and the target domain are the same task essentially
# therefore, models with the minimum validation loss will be adopted
# no matter it is generated in the source or the target domian
def eval_epoch_t(model, validation_data, type, opt):
    model.eval()

    def_all_train = []
    force_all_train = []
    force_all_true = []

    with torch.no_grad():
        total_loss = 0.0
        step_valid = 0
        for src_seq_v, trg_seq_v in validation_data:

            src_seq_v = src_seq_v.cuda().float()
            trg_seq_v = trg_seq_v.cuda().float()

            t_output = model(src_seq_v, src_seq_v)
            pred_seq_v = t_output[-1]
            loss = cal_loss(pred_seq_v, trg_seq_v)

            total_loss += loss.item() * src_seq_v.size(0)
            step_valid += src_seq_v.size(0)

            def_all_train.append(src_seq_v.detach().cpu().numpy())
            force_all_train.append(pred_seq_v.detach().cpu().numpy())
            force_all_true.append(trg_seq_v.detach().cpu().numpy())

    loss_average_1 = total_loss / step_valid

    def_all_train = np.concatenate(def_all_train, axis=0)
    force_all_train = np.concatenate(force_all_train, axis=0)
    force_all_true = np.concatenate(force_all_true, axis=0)

    io.savemat("./pred_result/un_data_%s_%s.mat" % (opt.folder, type),
                     {'y_train': force_all_train, 'y_target': force_all_true, 'X_train': def_all_train})

    return loss_average_1


def cal_loss(pred, real_value): # MSE Loss calculation
    loss_func = torch.nn.MSELoss(reduction="mean")
    loss = loss_func(pred, real_value)

    return loss

 # as long as both the source domain and the target domain are the same task essentially
# therefore, models with the minimum validation loss will be adopted
# no matter it is generated in the source or the target domian
def eval_epoch_s(model, validation_data, type, opt):
    model.eval()

    def_all_train = []
    force_all_train = []
    force_all_true = []

    with torch.no_grad():
        # for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
        total_loss = 0.0
        step_valid = 0
        for src_seq_v, trg_seq_v in validation_data:

            src_seq_v = src_seq_v.cuda().float()
            trg_seq_v = trg_seq_v.cuda().float()

            t_output = model(src_seq_v, src_seq_v)
            pred_seq_v = t_output[2]
            loss = cal_loss(pred_seq_v, trg_seq_v)

            total_loss += loss.item() * src_seq_v.size(0)
            step_valid += src_seq_v.size(0)

            def_all_train.append(src_seq_v.detach().cpu().numpy())
            force_all_train.append(pred_seq_v.detach().cpu().numpy())
            force_all_true.append(trg_seq_v.detach().cpu().numpy())

    loss_average_1 = total_loss / step_valid

    def_all_train = np.concatenate(def_all_train, axis=0)
    force_all_train = np.concatenate(force_all_train, axis=0)
    force_all_true = np.concatenate(force_all_true, axis=0)

    io.savemat("./pred_result/un_data_%s_%s.mat" % (opt.folder, type),
                     {'y_train': force_all_train, 'y_target': force_all_true, 'X_train': def_all_train})

    return loss_average_1


def main():

    opt = generate_para()
    all_models = ["model_test.chkpt"] # models to be loaded
    _, valid_t, test_t = load_data(opt)
    model = DAN.DANNet(opt).cuda() # if this model is replaced by the LSTM model, then it could be used for testing/validation of the LSTM model

    for file in all_models:
        print("Model name:", file, "\n", "dataset name:", opt.mat_file_s, opt.mat_file_t, opt.mat_file_t_test)
        opt.checkpoint_load = file
        opt.folder = opt.checkpoint_load.rstrip("chkpt").rstrip(".")
        checkpoint = torch.load("./model/%s" % opt.checkpoint_load, map_location="cuda:0")
        model.load_state_dict(checkpoint['net'])

        # loss_1 = eval_epoch(model, train_s, "un_4", opt)
        # testing results are also provided here, but it will not be considered in model selection
        # therefore, no leakage of the testing dataset will exist
        loss_2_s = eval_epoch_s(model, valid_t, "v_S10_s", opt)
        loss_2_t = eval_epoch_t(model, valid_t, "v_S10_t", opt)
        loss_3_s = eval_epoch_s(model, test_t, "t_S10_s", opt)
        loss_3_t = eval_epoch_t(model, test_t, "t_S10_t", opt)

        print(opt.folder, ":", loss_2_s, loss_2_t, loss_3_s, loss_3_t)



if __name__ == '__main__':
    main()
