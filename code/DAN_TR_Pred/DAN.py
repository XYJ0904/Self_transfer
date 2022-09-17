import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch
import argparse
import os


def generate_para():
    global global_step
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    opt.new_train = True
    opt.checkpoint_load = "AA.chkpt"

    opt.input_dim = 1
    opt.output_dim = 1
    opt.hidden_dim = 128

    opt.num_layers = 2
    opt.num_fc_layers = 1
    opt.dropout_ratio = 0.01
    opt.dropout_ratio_des = 0.01

    opt.seq_length = 2000

    return opt



class LSTM_model(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.number_layers = num_layers

        self.LSTM_1 = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.number_layers,
                              bias=True, batch_first=True)
        self.activeation = nn.ReLU()


    def forward(self, x):
        LSTM_output, _ = self.LSTM_1(x)
        LSTM_output = self.activeation(LSTM_output)

        return LSTM_output


class BT_network(nn.Module):

    def __init__(self, hidden_dim, output_dim, num_layers, dropout_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.number_layers = num_layers

        self.LSTM_1 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.number_layers,
                              bias=True, batch_first=True)
        self.act_LSTM = nn.ReLU()

        self.module_1 = nn.Sequential()

        self.module_1.add_module("linear_1", nn.Linear(hidden_dim, hidden_dim))
        self.module_1.add_module("act_1", nn.ReLU())
        self.module_1.add_module("dp_1", nn.Dropout(dropout_ratio))

        # self.Linear_2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        output = []

        output_1, _ = self.LSTM_1(x)
        output_1 = self.act_LSTM(output_1)

        output.append(output_1)

        output_2 = self.module_1(output_1)
        output.append(output_2)

        # output_3 = self.Linear_2(output_2)
        # output.append(output_3)

        return output


def generate_LSTM_model(opt, teacher=False):

    model = LSTM_model(opt.input_dim, opt.output_dim, opt.hidden_dim, opt.num_layers)

    if teacher:
        for param in model.parameters():
            param.detach_()

    return model


def generate_BT_model(opt, teacher=False):

    model = BT_network(opt.hidden_dim, opt.output_dim, opt.num_fc_layers, opt.dropout_ratio)

    if teacher:
        for param in model.parameters():
            param.detach_()

    return model


class DANNet(nn.Module):

    def __init__(self, args):
        super(DANNet, self).__init__()
        opt = generate_para()
        self.sharedNet = generate_LSTM_model(opt)
        self.bottleneck_1 = generate_BT_model(opt)
        self.bottleneck_2 = generate_BT_model(opt)

        self.linear_1 = nn.Linear(opt.hidden_dim, opt.output_dim)
        self.linear_2 = nn.Linear(opt.hidden_dim, opt.output_dim)


    def forward(self, source_data, target_data):
        LSTM_hidden_s = self.sharedNet(source_data)
        LSTM_mmd_s = self.bottleneck_1(LSTM_hidden_s)
        LSTM_output_s = self.linear_1(LSTM_mmd_s[-1])

        LSTM_hidden_t = self.sharedNet(target_data)

        LSTM_mmd_st = self.bottleneck_1(LSTM_hidden_t)
        LSTM_output_st = self.linear_1(LSTM_mmd_st[-1])
        # TODO: LSTM_output_st
        LSTM_mmd_t = self.bottleneck_2(LSTM_hidden_t)
        LSTM_output_t = self.linear_2(LSTM_mmd_t[-1])

        return LSTM_mmd_st, LSTM_output_st, LSTM_output_s, LSTM_mmd_t, LSTM_output_t