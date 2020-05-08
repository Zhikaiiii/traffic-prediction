import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        # number of features in input x
        self.input_size = input_size
        # number of features in hidden state h
        self.hidden_size = hidden_size
        # 输入门参数 Wi, Wc
        self.Wix = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Wih = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wcx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Wch = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # 遗忘门参数 Wf
        self.Wfx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Wfh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # 输出门参数 Wo
        self.Wox = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Woh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # 激活函数
        self.sigmoid_f = nn.Sigmoid()
        self.sigmoid_i = nn.Sigmoid()
        self.sigmoid_o = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()


    def forward(self, input):
        # 前向传播
        (x, h, c) = input
        it = self.sigmoid_i(self.Wix(x) + self.Wih(h))
        ft = self.sigmoid_f(self.Wfx(x) + self.Wfh(h))
        ot = self.sigmoid_o(self.Wox(x) + self.Woh(h))
        ct = ft*c + it*self.tanh_1(self.Wcx(x) + self.Wch(h))
        ht = ot*self.tanh_2(ct)
        return ct,ht

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_len, output_len, batch_size):
        super(MyLSTM, self).__init__()
        self.input_size = [input_size] + hidden_size
        self.hidden_size = hidden_size
        self.all_layers = []
        # LSTM的层数
        self.num_layers = num_layers
        # 输入的序列长度 即每层的cell个数
        self.input_len = input_len
        # 输出的序列长度
        self.output_len = output_len
        self.batch_size = batch_size
        for i in range(num_layers):
            cell = [LSTMCell(self.input_size[i], self.hidden_size[i])]
            self.all_layers.append(nn.Sequential(*cell))
        self.layer = nn.Sequential(*self.all_layers)
        self.hidden2out = nn.Linear(self.input_len, self.output_len)
        self.relu1 = nn.ReLU()
    # 前一层的输出h(l)是下一层的输入x(l+1)
    def forward(self,input):
        # 存储每层当前节点的值
        batch_size = input.shape[0]
        input_len = input.shape[1]
        feature_num = input.shape[2]
        state = []
        output = np.empty((batch_size,input_len,feature_num))
        output = torch.from_numpy(output)
        output = output.float()
        h_n = []
        c_n = []
        for i in range(input_len):
            x = input[:,i,:]
            for j in range(self.num_layers):
                # 第一个cell，初始化C和h
                if i == 0:
                    h,c = Variable(torch.zeros((batch_size,self.hidden_size[j]))), Variable((torch.zeros(batch_size,self.hidden_size[j])))
                    state.append((c, h))
                # 前向传播
                (c, h) = state[j]
                new_c, x = self.all_layers[j]((x, h, c))
                state[j] = (new_c, x)
                if i == self.input_len - 1:
                    h_n.append(x)
                    c_n.append(new_c)
            # 输出最后一层的h
            output[:,i,:] = x
        output = output.view(batch_size, -1, self.input_len)
        output = self.hidden2out(output)
        # output = self.relu1(output)
        output = output.view(batch_size, self.output_len, -1)
        return output, (h_n, c_n)

