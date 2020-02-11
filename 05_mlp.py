#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 全连接层
"""

import torch
import numpy as np
import sys
sys.path.append('.')
import d2lzh_pytorch as d2l
print(torch.__version__)

# 读取数据
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

# 定义模型参数
num_inputs,num_ouputs,num_hiddens=784,10,256

# h=w1x+b
# output=w2h+b
W1=torch.tensor(np.random.normal(0,0.01,(num_inputs,num_hiddens)),dtype=torch.float)
b1=torch.zeros(num_hiddens,dtype=torch.float)
W2=torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_ouputs)),dtype=torch.float)
b2=torch.zeros(num_ouputs,dtype=torch.float)

params=[W1,b1,W2,b2]
for param in params:
    param.requires_grad_(requires_grad=True)

# 定义激活函数
def relu(X):
    return torch.max(input=X,other=torch.tensor(0.0))


# 定义模型
def net(X):
    X=X.view(-1,num_inputs)
    H=relu(torch.matmul(X,W1)+b1)
    return relu(torch.matmul(H,W2)+b2)

# 定义损失函数
loss=torch.nn.CrossEntropyLoss()

num_epochs,lr=5,100.0

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)

