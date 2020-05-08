import data_process
from data_process import MyDataSet
import numpy as np
from baseline import SVR_baseline,evaluate,HA_baseline
from MyLSTM import  MyLSTM
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.utils.data as Data

adj, flow = data_process.load_data()
flow =np.mat(flow,dtype=np.float32)
# 读入邻接矩阵
adj_norm = data_process.cal_adj_norm(adj)

x_train, y_train, x_test, y_test = data_process.train_test_spilt(flow, 4, 20, 0.3)
y_pred = HA_baseline(x_test, y_test)
evaluate(y_test, y_pred)

# dataset
BATCH_SIZE = 32
train_dataset = MyDataSet(x_train, y_train, type = 'train')
test_dataset = MyDataSet(x_test, y_test, type = 'test')
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)


# LSTMmodel
input_size = x_train.shape[2]
hidden_size = [200, x_train.shape[2], x_train.shape[2]]
seq_len = 20
pre_len = 4
num_of_layers = 3
model = MyLSTM(input_size,hidden_size,num_of_layers,seq_len,pre_len,BATCH_SIZE)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
num_epochs = 50
train_loss = 0
model.train()  # 将网络转化为训练模式
for i in range(num_epochs):
    train_loader = tqdm(train_loader)
    train_loss = 0
    # scheduler.step()
    #     # lr = scheduler.get_lr()
    for j, (X, Y) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
        # X = X.view(-1,784)       #X:[64,1,28,28] -> [64,784]将X向量展平
        X = Variable(X)  # 转化数据类型
        # X = Variable(X)
        X = X.float()
        Y = Variable(Y)
        Y = Y.float()
        # label = Variable(label)
        out,_ = model(X)
        # out = out[0][:,seq_len-pre_len:seq_len,:] # 正向传播
        lossvalue = loss(out, Y)  # 求损失值
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值
        optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数
        # 计算损失
        train_loss += float(lossvalue)
    print("train epoch:" + ' ' + str(i))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))

model.eval() #模型转化为评估模式
test_loss = 0
y_pred = None
for i,(X, Y) in enumerate(test_loader):
        X = Variable(X)
        X = X.float()
        Y = Variable(Y)
        Y = Y.float()
        with torch.no_grad():
            testout,_ = model(X)
        lossvalue = loss(testout,Y)
        if y_pred is None:
            y_pred = torch.cat([testout])
        else:
            y_pred = torch.cat([y_pred, testout])
        # 计算损失
        test_loss += float(lossvalue)
y_pred = y_pred.numpy()
evaluate(y_test,y_pred)
print("lose:" + ' ' + str(test_loss / len(test_loader)))