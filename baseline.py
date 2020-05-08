# coding=utf-8
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from tqdm import tqdm

def HA_baseline(x_test,y_test):
    # 节点个数
    num_nodes = x_test.shape[2]
    pred_len = y_test.shape[1]
    y_pred_all = np.empty(y_test.shape)
    for i in tqdm(range(num_nodes)):
        x_tmp = x_test[:,:,i]
        y_pred = np.mean(x_tmp, axis=1)
        y_pred = np.repeat(y_pred, pred_len).reshape([-1, pred_len])
        y_pred_all[:,:,i] = y_pred
    return y_pred_all


def SVR_baseline(x_train, y_train, x_test, y_test):
    # 节点个数
    num_nodes = x_train.shape[2]
    pred_len = y_train.shape[1]
    model = SVR(kernel='linear')
    y_pred_all = np.empty(y_test.shape)
    for i in tqdm(range(num_nodes)):
        y_tmp = y_train[:,:,i]
        x_tmp = x_train[:,:,i]
        # 将y值改为一维
        y_tmp = np.mean(y_tmp, axis=1)
        model.fit(x_tmp, y_tmp)
        y_pred = model.predict(x_test[:, :, i])
        y_pred = np.repeat(y_pred, pred_len).reshape([-1, pred_len])
        y_pred_all[:,:,i] = y_pred
    return y_pred_all

# evaluate the model
def evaluate(y_test, y_pred):
    (test_num,pred_len,node_num) = y_pred.shape
    loss = 0
    for i in range(test_num):
        for j in range(pred_len):
            for k in range(node_num):
                loss += (y_test[i,j,k] - y_pred[i,j,k])**2
    loss = loss/(test_num*node_num*pred_len)
    y_pred = y_pred.reshape([1,-1])
    y_test = y_test.reshape([1,-1])
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("mean squared error: %0.3f" %mse)
    print("mean absolute error: %0.3f" %mae)
    print("mean absolute error: %0.3f" %loss)
