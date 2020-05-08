import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Graph_convolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, activation='relu'):
        super(Graph_convolution, self).__init__()
        self.bias = None
        # 权重矩阵
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        # 激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()
    def forward(self, x, adj):
        out = torch.mm(adj, x)
        out = torch.mm(out, self.weight)
        if self.bias:
            out += self.bias
        return self.act(out)

class GCN(nn.module):
    def __init__(self, input_dim, output_dim,adj):
        super(GCN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.layer1 = Graph_convolution(self.input_dim, 100)
        self.layer2 = Graph_convolution(100, self.output_dim, activation='sigmoid')

    def forward(self, x):
        out = self.layer1(x, self.adj)
        out = self.layer2(out, self.adj)
        return out
