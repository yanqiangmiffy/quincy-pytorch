#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 参数
"""
import torch
from torch import nn
from torch.nn import init

# 模型参数的访问、初始化和共享

print(torch.__version__)

net=nn.Sequential(nn.Linear(4,3),
                  nn.ReLU(),
                  nn.Linear(3,1))
print(net)

X=torch.rand(2,4)
Y=net(X).sum()

# 访问模型参数
print(type(net.parameters()))
for name,param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param,mean=0,std=0.01)
        print(name,param.data)

for name,param in net.named_parameters():
    if 'bias' in name:
        param.data+=1
        print(name,param.data)


linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)

print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)
