import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as Data


def load_data():
    adj_matrix = pd.read_csv('data/sz_adj.csv',header=None)
    adj = np.mat(adj_matrix)
    flow = pd.read_csv('data/sz_speed.csv')
    return adj, flow


# 计算邻接矩阵
def cal_adj_norm(adj):
    node_num = adj.shape[0]
    adj = np.asarray(adj)
    adj_ = adj + np.eye(node_num)
    # 度矩阵
    d = np.sum(adj_,1)
    d_sqrt = np.power(d, -0.5)
    d_sqrt = np.diag(d_sqrt)
    adj_norm = np.dot(np.dot(d_sqrt, adj_), d_sqrt)
    return adj_norm

# data spilt
#  pred_len : 输出的序列长度
#  train_len: 预测的序列长度
def train_test_spilt(data, pred_len, train_len, test_percentage):
    data_size = data.shape[0]
    test_size = int(data_size*test_percentage)
    train_size = data_size - test_size
    # 训练集和测试集划分
    data_train = data[0: train_size]
    data_test = data[train_size:]
    x_train, y_train, x_test, y_test = [],[],[],[]
    # 训练集
    for i in range(train_size - train_len - pred_len):
        x_train.append(data_train[i: i + train_len])
        y_train.append(data_train[i + train_len: i + train_len + pred_len])
    # 测试集
    for i in range(test_size - train_len - pred_len):
        x_test.append(data_test[i: i + train_len])
        y_test.append(data_test[i + train_len: i + train_len + pred_len])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


class MyDataSet(Data.Dataset):
    def __init__(self, x, y, type = 'train'):  # 第一步初始化各个变量
        self.x = x #导入所有数据
        self.y = y
        self.type = type
    def __getitem__(self, idx):  # 获取数据
        x = self.x[idx]
        y = self.y[idx]
        return x,y
    def __len__(self):
        return len(self.x)  # 返回数据集长度