#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description:线性回归
"""
import torch
from time import time
print(torch.__version__)

a=torch.ones(1000)
b=torch.ones(1000)

# 将这两个向量按元素逐一做标量加法:
start=time()
c=torch.zeros(1000)
for i in range(1000):
    c[i]=a[i]+b[i]
print(time()-start)

# 两个向量直接相加
start=time()
d=a+b
print(time()-start)

# 广播机制例子🌰：
a=torch.ones(3)
b=10
print(a+b)

###################################33
# 线性回归的简介实现
import torch
from torch import nn
import numpy as np

print(torch.__version__)

# 生成数据集
num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2

features=torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+b
# 添加噪音
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据
import torch.utils.data as Data
batch_size=10

# 将训练数据的特征与标签组合
datastet=Data.TensorDataset(features,labels)

# 将dataset放入DataLoader
data_iter=Data.DataLoader(
    dataset=datastet,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
for X,y in data_iter:
    print(X,'\n',y)
    break

# 定义模型
class LinearNet(nn.Module):
    def __init__(self,n_features):
        """
        初始化
        :param n_features: 特征个数，输入维度
        """
        super(LinearNet, self).__init__()
        self.linear=nn.Linear(n_features,1)
    def forward(self,x):
        y=self.linear(x)
        return y
net=LinearNet(num_inputs)
print(net)

# =============== 其他写法 ===================

# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])


for param in net.parameters():
    print(param)

# 初始化模型参数
from torch.nn import init
init.normal_(net[0].weight,mean=0.0,std=0.1)
init.constant_(net[0].bias,val=0.0)

for param in net.parameters():
    print(param)

# 定义损失函数
loss=nn.MSELoss()
# 定义优化算法
import  torch.optim as optim
optimizer=optim.SGD(net.parameters(),lr=0.03)
print(optimizer)


# 为不同子网络设置不同的学习率
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)

# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

# 训练模型
num_epochs=3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        output=net(X)
        l=loss(output,y.view(-1,1))
        optimizer.zero_grad() # 梯度清零，等价于net.zeo_grad()
        l.backward()
        optimizer.step()
    format='epoch %d, loss: %f'
    print(format % (epoch,l.item()))

dense=net[0]
print(true_w,dense.weight.data)
print(true_b,dense.bias.data)

# [2, -3.4] tensor([[ 1.9994, -3.3995]])
# 4.2 tensor([10.0001])