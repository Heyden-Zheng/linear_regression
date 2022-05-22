import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from model import LinearRegression

"""
步骤：
    1.设置线性回归模型的参数
    2.加载数据集
    3.定义损失函数
    4.定义优化器（优化函数）
    5.训练模型
    6.测试模型
"""

# 1.1设置x和y的维度
input_size = 1
output_size = 1

# 1.2设置学习率
learning_rate = 0.001

# 2.1 加载训练数据集
train_dataset = pd.read_csv('./data/train_data.csv', encoding='utf-8').astype({'x': 'float32', 'y': 'float32'})
x_train = np.array(train_dataset[['x']])
y_train = np.array(train_dataset[['y']])
# 2.2 加载测试数据集
test_dataset = pd.read_csv('./data/test_data.csv', encoding='utf-8').astype({'x': 'float32', 'y': 'float32'})
x_test = np.array(test_dataset[['x']])
y_test = np.array(test_dataset[['y']])

# 画散点图，观察数据分布情况
plt.figure()  # 生成画板
plt.scatter(x_train, y_train)
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()

# 3.定义损失函数
criterion = nn.MSELoss()  # 总平方误差，即模型对每个训练样本的预测值与真实值做差的平方的总和

# 4.定义优化函数
#   4.1 优化器代表我们需要通过什么方式优化需要学习的值，对于线性回归模型来说，指的就是w和b
#   4.2 lr指学习效率，一般是小于1的数，值越小模型收敛越慢，但是值过大的话，模型可能无法收敛
model = LinearRegression(input_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 5.训练模型
def train():
    num_epochs = 1000  # 模型训练1000次
    for epoch in range(num_epochs):
        inputs = Variable(torch.Tensor(x_train))
        targets = Variable(torch.Tensor(y_train))  # Variable是一个可以变化的变量，符合反向传播参数更新的特性
        # 每次反向传播的时候需要将参数的梯度重置为0
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 根据预测值和真实值计算损失值
        loss = criterion(outputs, targets)
        # 通过反向传播实现训练参数(w和b)的更新
        loss.backward()
        optimizer.step()
        # 每隔50次打印一次结果
        if (epoch + 1) % 50 == 0:
            print('Epoch[%d/%d], Loss:%.4f' % (epoch + 1, num_epochs, loss.item()))
    print('======模型训练完毕======')


# 测试模型
def test():
    # 测试模型效果
    model.eval()  # 开启评估模式
    print('========开始测试========')
    inputs = Variable(torch.Tensor(x_test))
    targets = Variable(torch.Tensor(y_test))  # 将Numpy类型的y_test转换成Tensor，后面损失函数要求参数为Tensor类型
    predicted = model(inputs)
    test_loss = criterion(predicted, targets)
    print('Test set results:', 'loss = {:.4f}'.format(test_loss.item()))

    plt.plot(x_test, y_test, 'ro')
    plt.plot(x_test, predicted.detach(), label='predict')
    plt.legend()
    plt.xlabel('x_test')
    plt.ylabel('y_test')
    plt.show()


# 开始训练
train()

# 开始测试
test()
