from sklearn.datasets import load_boston
from sklearn import preprocessing
import numpy as np

# 载入数据，并预处理
X, y = load_boston(return_X_y=True)
X = preprocessing.scale(X[:100, :])
y = preprocessing.scale(y[:100].reshape(-1, 1))

# 定义超参数
data_size, D_input, D_output, D_hidden = X.shape[0], X.shape[1], 1, 50
lr = 1e-5
epoch = 200000

w1 = np.random.randn(D_input, D_hidden)
w2 = np.random.randn(D_hidden, D_output)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(epoch):

    # 前向传播
    h = np.dot(X, w1)
    h_ = sigmoid(h)
    y_ = np.dot(h, w2)

    # 打印误差
    mse = np.sum((y - y_) ** 2)
    if i % 10 == 0:
        print('epoch: {} loss: {:.4f}'.format(i, mse))

    # 误差反向传播
    g_y_ = 2 * (y_ - y)
    g_w2 = np.dot(h_.T, g_y_)
    g_h_ = np.dot(g_y_, w2.T)
    g_h = g_h_ * sigmoid(h) * (1 - sigmoid(h))
    g_w1 = np.dot(X.T, g_h)

    # 参数更新
    w1 -= lr * g_w1
    w2 -= lr * g_w2
