# Readme

## base.py

base.py中包含torch.tensor，numpy.array，以及一般list在使用方式上的区别，

重点强调了torch.tensor的自动求导机制。

## base_network.py

base_network.py包含一个神经网络拟合函数的定义方式，内部的参数结构，Loss计算方式，优化方法的组织形式。

一般顺序是先定义网络，再构造loss函数，并用优化方法将需要优化的参数联系起来。

然后计算在样本点的loss值，以及各个参数在样本点对应的梯度，最后用对应优化方法更新参数。

这个文件没有使用任何数据集。

