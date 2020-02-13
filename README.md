# quincy-pytorch
pytorch学习笔记

## 学习资料
- [Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch)
- [伯禹教育-ElitesAI·动手学深度学习PyTorch版](https://www.boyuai.com/elites/course/cZu18YmweLv10OeV)

## 知识点

- 协变量偏移
```text
这里我们假设，虽然输入的分布可能随时间而改变，但是标记函数，即条件分布P（y∣x）不会改变。虽然这个问题容易理解，但在实践中也容易忽视。

想想区分猫和狗的一个例子。我们的训练数据使用的是猫和狗的真实的照片，但是在测试时，我们被要求对猫和狗的卡通图片进行分类。
```
[梯度消失、梯度爆炸](https://www.boyuai.com/elites/course/cZu18YmweLv10OeV/jupyter/0-S8U3RMs-8Eq2qWsGSzF)

