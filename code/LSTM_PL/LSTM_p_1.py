import argparse
import math
import time
import numpy as np
import random as rd
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.autograd import Variable
from Models import LSTM_model
from utils_BG import *

global_step = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def generate_para():
    global global_step
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.log = True
    opt.save_mode = 'best'

    # dataset is organized in the format of .mat through the scipy.io.savemat/loadmat
    opt.root_path_T = "./" # absolute/relative path of training dataset
    opt.mat_file_T = "data_huang_norm_final.mat" # file name of training dataset, usually the pesudo-labeled dataset
    opt.root_path_V = "./" # absolute/relative path of validation dataset
    opt.mat_file_V = "data_huang_norm_final.mat" # file name of validation dataset, usually the labeled dataset
    opt.key_X_train = "X_train" # key of data in the .mat file
    opt.key_y_train = "y_train"
    opt.key_X_valid = "X_valid"
    opt.key_y_valid = "y_valid"

    opt.cuda = True
    opt.batch_size = 10 # batch_size
    opt.LR = 2e-3 # learnning rate

    opt.epoch = 1000 # number of epoches
    opt.series = "LSTM_320_1" # file name to be saved

    print("Training model with series %s" % opt.series)
    opt.random_seed = "1"

    opt.log_name = "./log/log_%s" % opt.series
    opt.save_model_name = "./model/model_%s" % opt.series

    if not os.path.exists("./log"):
        os.mkdir("./log")
    if not os.path.exists("./model"):
        os.mkdir("./model")

    return opt


def generate_random_noise(size_src): # generate random noise for teacher model
    # print(size_src)
    random_noise = torch.rand(size_src)
    random_noise = random_noise.cuda()
    return random_noise


def update_model_T(model_S, model_T, alpha, global_step): # update the teacher model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step * 10 + 1), alpha)
    for param_T, param_S in zip(model_T.parameters(), model_S.parameters()):
        param_T.data.mul_(alpha).add_(1 - alpha, param_S.data)


def train_epoch(model_S, model_T, training_data, optimizer, decay_optim, opt, epoch):
    ''' Epoch operation in training phase'''

    model_S.train()

    loss_pred_all, loss_sta_all = 0, 0

    step = 0
    for src_seq, trg_seq in training_data:

        src_seq = src_seq.cuda().float()
        trg_seq = trg_seq.cuda().float()

        optimizer.zero_grad()

        noise = torch.rand(*src_seq.size()) * 4e-3 - 2e-3 # noised input
        noise = noise.cuda().float()
        src_seq_noise = src_seq + noise

        pred_labeled = model_S(src_seq)
        pred_labeled_noise = model_T(src_seq_noise)  # forward propogation of teacher model

        pred_loss = cal_loss(pred_labeled, trg_seq)
        sta_loss = cal_loss(pred_labeled, pred_labeled_noise) # consistency loss

        loss = pred_loss + 0.1 * sta_loss # weight of consistency loss could be adjusted according to your task

        loss.backward()
        optimizer.step()

        loss_pred_all += pred_loss.item() * src_seq.size(0)
        loss_sta_all += sta_loss.item() * src_seq.size(0)
        step += src_seq.size(0)

        update_model_T(model_S, model_T, 0.999, epoch)

    loss_pred_average = loss_pred_all / step
    loss_sta_average = loss_sta_all / step

    return loss_pred_average, loss_sta_average


def eval_epoch(model, validation_data):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    with torch.no_grad():
        # for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
        total_loss = 0.0
        step_valid = 0
        for src_seq_v, trg_seq_v in validation_data:

            src_seq_v = src_seq_v.cuda().float()
            trg_seq_v = trg_seq_v.cuda().float()

            pred_seq_v = model(src_seq_v)
            # pred_seq_v = pred_seq_v.unsqueeze(2)
            loss = cal_loss(pred_seq_v, trg_seq_v)

            total_loss += loss.item() * src_seq_v.size(0)
            step_valid += src_seq_v.size(0)

    loss_average_1 = total_loss / step_valid

    return loss_average_1


def train(model_S, model_T, training_data, validation_data, optimizer, decay_optim, device, opt, start_epoch):
    ''' Start training '''

    if opt.log:
        log_file = opt.log_name + '.log'
        print('[Info] Training performance will be written to file: {}'.format(log_file))
        with open(log_file, 'w') as log_f:
            log_f.write('epoch, train_loss, valid_loss\n')

    def print_performances(header, loss, start_time):
        print('  - {header:15} : {loss: 8.10f}, '\
              'elapse: {elapse:3.3f} min'.format(header=f"({header})", loss=loss,
                                                 elapse=(time.time()-start_time)/60))

    valid_losses = []
    valid_losses_strict = []

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss_pred, train_loss_sta = train_epoch(model_S, model_T, training_data, optimizer, decay_optim, opt, epoch_i)
        print_performances('Training Loss Pred', train_loss_pred * 1000, start) # loss * 1000 is designed for clear screen print, without any influcence on the final result
        print_performances('Training Loss Sta', train_loss_sta * 1000, start)
        decay_optim.step()

        start = time.time()
        valid_loss = eval_epoch(model_S, validation_data)
        print_performances('Validation Loss Pred', valid_loss * 1000, start)
        # print_performances('Validation Loss Strict', valid_loss_strict, start)

        valid_losses += [valid_loss]
        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model_S.state_dict()}

        if opt.save_mode == 'all':
            model_name = opt.save_model_name + '_loss_{valid_loss:3.3f}.chkpt'.format(valid_loss=valid_loss)
            torch.save(checkpoint, model_name)
        elif opt.save_mode == 'best':
            model_name = opt.save_model_name + '_best.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_file, 'a') as log_f:
            log_f.write('{epoch},{train_loss: 8.10f},{valid_loss: 8.10f},\n'.format(
                epoch=epoch_i, train_loss=train_loss_pred, valid_loss=valid_loss))



def main():

    opt = generate_para()

    device = torch.device('cuda' if opt.cuda else 'cpu')
    training_data, validation_data, unlabeled_data = prepare_dataloaders(opt)
    model = LSTM_model() # parameter of the LSTM model is provided in another "Models"
    model = torch.nn.DataParallel(model).cuda()

    model_T = copy.deepcopy(model) # generate teacher model and detach the gradient
    model_T = torch.nn.DataParallel(model_T).cuda()

    for param in model_T.parameters():
        param.detach_()

    optimizer = optim.Adam(model.parameters(), lr=opt.LR, betas=(0.9, 0.99), eps=1e-09)
    decay_optim = optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

    train(model, model_T, training_data, validation_data, optimizer, decay_optim, device, opt, 0)


if __name__ == '__main__':
    main()
