#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 存储和写入模型
"""
import torch
from torch import nn
print(torch.__version__)

# 读写Tensor
x = torch.ones(3)
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
torch.save({'x': x, 'y': y}, 'xy_dict.pt')
xy = torch.load('xy_dict.pt')
