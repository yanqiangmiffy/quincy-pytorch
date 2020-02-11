#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 

"""
import torch

torch.manual_seed(0)

torch.cuda.manual_seed(0)
print(torch.__version__)

# 测试GPU是否可用
print(torch.cuda.is_available())

# 创建一个5x3的未初始化的Tensor：
x = torch.empty(5, 3)
print(x)

# 创建一个5x3的随机初始化的Tensor:
x = torch.rand(5, 3)
print(x)

# 创建一个5x3的long型全0的Tensor:
x=torch.zeros(5,4,dtype=torch.long)
print(x)

# 直接根据数据创建:
x=torch.tensor([5.5,3])
print(x)


# 还可以通过现有的Tensor来创建，此方法会默认重用输入Tensor的一些属性，例如数据类型，除非自定义数据类型。


x = x.new_ones(5, 3, dtype=torch.float64)      # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 指定新的数据类型
print(x)

# 我们可以通过shape或者size()来获取Tensor的形状:

print(x.size())
print(x.shape)

# 操作

# 算术操作
# 加法形式1
y = torch.rand(5, 3)
print(x + y)
# 加法形式2
print(torch.add(x, y))

# 输出结果
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 加法形式3 inplace
y.add_(x)
print(y)


# 索引
# 我们还可以使用类似NumPy的索引操作来访问Tensor的一部分，需要注意的是：索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改。

y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了


# 改变形状
y=x.view(15)
z=x.view(-1,5) # -1代表可以根据其他维度（5）推算出来真实维度(3)
print(x.size(),y.size(),z.size())

# 注意view()返回的新tensor与源tensor共享内存，也即更改其中的一个，另外一个也会跟着改变。
x += 1
print(x)
print(y) # 也加了1

# 如果不想共享内存，推荐先用clone创造一个副本然后再使用view。
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)

# 另外一个常用的函数就是item(), 它可以将一个标量Tensor转换成一个Python number：
x = torch.randn(1)
print(x)
print(x.item())

# 广播机制

x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

# 运算的内存开销
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False


x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before) # True

# Tensor与NumPy的转换

a = torch.ones(5)
b = a.numpy()
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

# Numpy转为Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)

a += 1
print(a, b)
b += 1
print(a, b)

# 直接用torch.tensor()将NumPy数组转换成Tensor，该方法总是会进行数据拷贝，返回的Tensor和原来的数据不再共享内存。

 # 用torch.tensor()转换时不会共享内存
c = torch.tensor(a)
a += 1
print(a, c)


# Tensor On GPU
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型


