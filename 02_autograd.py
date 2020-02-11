#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 自动求梯度

"""
import torch
print(torch.__version__)

# Function是另外一个很重要的类。Tensor和Function互相结合就可以构建一个记录有整个计算过程的非循环图。每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function（除非用户创建的Tensors时设置了grad_fn=None）。

# Tensor
x=torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)

y=x+2
print(y)
print(y.grad_fn)

# 注意x是直接创建的，所以它没有grad_fn, 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn。
print(x.is_leaf,y.is_leaf)

z = y * y * 3
out=z.mean()
print(z,out)
# 通过.requires_grad_()来用in-place的方式改变requires_grad属性：


a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False

a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)

# 梯度
out.backward() # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

