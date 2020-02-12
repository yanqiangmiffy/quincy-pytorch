#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 使用GPU
"""
import torch
from torch import nn
print(torch.__version__)

# 计算设备
# 判断cuda是否可用
print(torch.cuda.is_available())
# gpu数量
print(torch.cuda.device_count())
# 当前设备索引，从0开始
print(torch.cuda.current_device())
# 查看 gpu的名字
print(torch.cuda.get_device_name(0))


# Tensor的GPU计算
x=torch.tensor([1,2,3])
print(x)

x=x.cuda(0)
print(x)
print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
print(x)

y = x**2
print(y)

# 模型的GPU计算
net=nn.Linear(3,1)
print(list(net.parameters())[0].device)
net.cuda()
print(list(net.parameters())[0].device)

x = torch.rand(2,3).cuda()
net(x)
