import argparse
import datetime
import os
import time

import torch.cuda
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score
from torch import nn
from torch.utils.data import DataLoader

from data import Data
from model import LSKT, LSKTSK, LSKTC, LSKTNS, LSKTNT
from utils import *

parser = argparse.ArgumentParser(description='Script to test DKT.')
import configparser

cf = configparser.ConfigParser()
cf.read("config.ini")
section = 'assist2009_updated'
parser.add_argument('--dataset', type=str, default=cf.get(section, 'dataset'), help='')
parser.add_argument('--m', type=int, default=cf.getint(section, 'm'), help='')
parser.add_argument('--n', type=int, default=cf.getint(section, 'n'), help='')
parser.add_argument('--learning_rate', type=float, default=cf.getfloat(section, 'learning_rate'), help='')
parser.add_argument('--kernel_size', type=int, default=cf.getint(section, 'kernel_size'), help='')
parser.add_argument('--length', type=int, default=cf.getint(section, 'length'), help='')
parser.add_argument('--epochs', type=int, default=cf.getint(section, 'epochs'), help='')
parser.add_argument('--batch_size', type=int, default=cf.getint(section, 'batch_size'), help='')
parser.add_argument('--seed', type=int, default=cf.getint(section, 'seed'), help='')
parser.add_argument('--q_num', type=int, default=cf.getint(section, 'q_num'), help='')
parser.add_argument('--cv_num', type=int, default=cf.getint(section, 'cv_num'), help='')
parser.add_argument('--data_path', type=str, default=cf.get(section, 'data_path'), help='')
parser.add_argument('--opt', type=str, default=cf.get(section, 'opt'), help='')
parser.add_argument('--ffn_h_num', type=int, default=cf.getint(section, 'ffn_h_num'), help='')
parser.add_argument('--d_model', type=int, default=cf.getint(section, 'd_model'), help='')
parser.add_argument('--encoder_out', type=int, default=cf.getint(section, 'encoder_out'), help='')
parser.add_argument('--dropout', type=float, default=cf.getfloat(section, 'dropout'), help='')
parser.add_argument('--channel_size', type=int, default=cf.getint(section, 'channel_size'), help='')
parser.add_argument('--d_ff', type=int, default=cf.getint(section, 'd_ff'), help='')
parser.add_argument('--model_type', type=str, default='lsktnt', help='')
params = parser.parse_args()
dataset = params.dataset
if dataset == 'junyi3':
    params.q_num = 713
if dataset == 'poj3':
    params.q_num = 4055
if dataset == 'eanalyst_math':
    params.q_num = 2696
if dataset == 'assist2015':
    params.q_num = 104
if dataset == 'assist2009_updated':
    params.q_num = 112
if dataset == 'STATICS':
    params.q_num = 1224
if dataset == 'statics':
    params.q_num = 1224


def collate(batch, q_num=params.q_num, length=params.length):
    # lens = [len(row) for row in batch]
    max_len = length  # max(lens)
    batch = tensor([[[*e, 1] for e in row] + [[0, 0, 0]] * (max_len - len(row)) for row in batch])
    Q, Y, S = batch.T
    Q, Y, S = Q.T, Y.T, S.T  # torch.size([32,200])
    X = Q + q_num * (1 - Y)
    return X, Y, S, Q


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataPrefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


def train(model, data, optimizer, batch_size):
    model.train(mode=True)
    # 创建一个测量目标和输出之间的二进制交叉熵的标准
    criterion = nn.BCELoss()
    q_onehot = eye(data.q_num)
    data_loader = DataLoaderX(
        dataset=data,
        batch_size=batch_size,
        collate_fn=collate,  # (batch, data.q_num, params.length),
        shuffle=True,
        # num_workers=2
    )
    for X, Y, S, Q in data_loader:
        x = Q.contiguous()
        y = Y.contiguous()
        P = model(x, y)
        Q = q_onehot[Q]
        Q, P, Y, S = Q[:, 1:], P[:, :-1], Y[:, 1:], S[:, 1:]
        P = (Q * P).sum(2)
        index = S == 1
        loss = criterion(P[index], Y[index].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, data, batch_size):
    model.eval()
    criterion = nn.BCELoss()
    y_pred, y_true = [], []
    loss = 0.0
    q_onehot = eye(data.q_num)
    data_loader = DataLoaderX(
        dataset=data,
        batch_size=batch_size,
        collate_fn=collate,  # lambda batch: collate(batch, data.q_num, params.length),
        shuffle=True,
    )
    for X, Y, S, Q in data_loader:
        x = Q.contiguous()
        y = Y.contiguous()
        P = model(x, y)
        Q = q_onehot[Q]
        Q, P, Y, S = Q[:, 1:], P[:, :-1], Y[:, 1:], S[:, 1:]
        P = (Q * P).sum(2)

        index = S == 1
        P, Y = P[index], Y[index].float()
        y_pred += detach(P)
        y_true += detach(Y)
        loss += detach(criterion(P, Y) * P.shape[0])

    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    # auc, loss, mse, acc
    return auc(fpr, tpr), loss / len(y_true), mse_value, mae_value, acc_value


def experiment(data_path, dataset, m, n, learning_rate, length, kernel_size, epochs, batch_size, seed, q_num,
               cv_num, ffn_h_num, opt, d_model, encoder_out, dropout, channel_size, d_ff, model_type='lskt'):
    set_seed(seed)
    path = './result_dkt_cross/%s' % ('{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()))
    os.makedirs(path)
    info_file = open('%s/info.txt' % path, 'w+')
    params_list = (
        'dataset = %s\n' % dataset,
        'm = %d\n' % m,
        'n = %d\n' % n,
        'learning_rate = %f\n' % learning_rate,
        'length = %d\n' % length,
        'kernel_size = %d\n' % kernel_size,
        'batch_size = %d\n' % batch_size,
        'seed = %d\n' % seed,
        'q_num = %d\n' % q_num
    )
    info_file.write('file_name = allxt-onehot no norm + weight decay 5e-4')
    info_file.write('%s%s%s%s%s%s%s%s%s' % params_list)
    model_list = []

    for cv in [cv_num]:
        random.seed(cv + 1000)
        random.seed(0)

        train_data = Data(open('%s/%s/%s_train%d.csv' % (data_path, dataset, dataset, cv), 'r'), length, q_num,
                          is_test=True)
        valid_data = Data(open('%s/%s/%s_valid%d.csv' % (data_path, dataset, dataset, cv), 'r'), length, q_num,
                          is_test=True)
        max_auc = 0.0
        # DKT模型实例化：model
        if model_type == 'lskt':
            model = cuda(
                LSKT(kernel_size=kernel_size, num_channels=[channel_size] * m, q_num=q_num, d_model=d_model,
                     encoder_out=encoder_out, dropout=dropout, ffn_h_num=ffn_h_num, d_ff=d_ff, n=n))
        if model_type == 'lsktns':
            model = cuda(
                LSKTNS(kernel_size=kernel_size, num_channels=[channel_size] * m, q_num=q_num, d_model=d_model,
                       encoder_out=encoder_out, dropout=dropout, ffn_h_num=ffn_h_num, d_ff=d_ff, n=n))
        if model_type == 'lsktsk':
            model = cuda(
                LSKTSK(kernel_size=kernel_size, num_channels=[channel_size] * m, q_num=q_num, d_model=d_model,
                       encoder_out=encoder_out, dropout=dropout, ffn_h_num=ffn_h_num, d_ff=d_ff, n=n))
        if model_type == 'lsktnt':
            model = cuda(
                LSKTNT(kernel_size=kernel_size, num_channels=[channel_size] * m, q_num=q_num, d_model=d_model,
                       encoder_out=encoder_out, dropout=dropout, ffn_h_num=ffn_h_num, d_ff=d_ff, n=n))
        if opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        early_stop = 0
        for epoch in range(1, epochs + 1):
            time_start = time.time()
            train(model, train_data, optimizer, batch_size)
            train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model, train_data, batch_size)
            valid_auc, valid_loss, valid_mse, valid_mae, valid_acc = evaluate(model, valid_data, batch_size)
            time_end = time.time()
            if max_auc < valid_auc:
                early_stop = epoch
                max_auc = valid_auc
                torch.save(model.state_dict(), '%s/model_%s' % ('%s' % path, '%d' % cv))
                current_max_model = model
            if epoch - early_stop > 30:
                break

            print_list = (
                'cv:%-3d' % cv,
                'epoch:%-3d' % epoch,
                'max_auc:%-8.4f' % max_auc,
                'valid_auc:%-8.4f' % valid_auc,
                'valid_loss:%-8.4f' % valid_loss,
                'valid_mse:%-8.4f' % valid_mse,
                'valid_mae:%-8.4f' % valid_mae,
                'valid_acc:%-8.4f' % valid_acc,
                'train_auc:%-8.4f' % train_auc,
                'train_loss:%-8.4f' % train_loss,
                'train_mse:%-8.4f' % train_mse,
                'train_mae:%-8.4f' % train_mae,
                'train_acc:%-8.4f' % train_acc,
                'time:%-6.2fs' % (time_end - time_start)
            )

            print('%s %s %s %s %s %s %s %s %s %s %s %s %s %s' % print_list)
            info_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s\n' % print_list)
        model_list.append(current_max_model)

    train_list = []
    auc_list = []
    mse_list = []
    mae_list = []
    acc_list = []
    loss_list = []
    for _, model_item in enumerate(model_list):
        train_data = Data(open('%s/%s/%s_train%d.csv' % (data_path, dataset, dataset, cv_num), 'r'), length, q_num,
                          is_test=True)
        test_data = Data(open('%s/%s/%s_test%d.csv' % (data_path, dataset, dataset, 0), 'r'), length, q_num,
                         is_test=True)
        train_auc, train_loss, train_mse, train_mae, train_acc = evaluate(model_item, train_data, batch_size)
        test_auc, test_loss, test_mse, test_mae, test_acc = evaluate(model_item, test_data, batch_size)

        train_list.append(train_auc)
        auc_list.append(test_auc)
        mse_list.append(test_mse)
        mae_list.append(test_mae)
        acc_list.append(test_acc)
        loss_list.append(test_loss)
        print_list_test = (
            'cv:%-3d' % cv_num,
            'train_auc:%-8.4f' % train_auc,
            'test_auc:%-8.4f' % test_auc,
            'test_mse:%-8.4f' % test_mse,
            'test_mae:%-8.4f' % test_mae,
            'test_acc:%-8.4f' % test_acc,
            'test_loss:%-8.4f' % test_loss
        )

        print('%s %s %s %s %s %s %s\n' % print_list_test)
        info_file.write('%s %s %s %s %s %s %s\n' % print_list_test)

    info_file.write('%s %s %s %s %s %s\n' % print_result)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    for cv in range(params.cv_num):
        # params.cv_num = cv
        print(params)
        experiment(
            data_path=params.data_path,
            dataset=params.dataset,
            m=params.m,
            n=params.n,
            learning_rate=params.learning_rate,
            length=params.length,
            epochs=params.epochs,
            batch_size=params.batch_size,
            seed=params.seed,
            q_num=params.q_num,
            cv_num=params.cv_num,
            kernel_size=params.kernel_size,
            ffn_h_num=params.ffn_h_num,
            opt=params.opt,
            d_model=params.d_model,
            encoder_out=params.encoder_out,
            dropout=params.dropout,
            channel_size=params.channel_size,
            d_ff=params.d_ff,
            model_type=params.model_type
        )
