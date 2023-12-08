import torch
print(torch.__version__)

import torch.nn as nn

# F中具有的常用函数一般不具备可学习参数
import torch.nn.functional as F

from base import my_print

class Net(nn.Module):
    """
    继承torch.nn.Module，在最简单的神经网络中，我们只考虑卷积层和全连接层
    """
    def __init__(self) -> None:
        # 必须要先初始化父类
        super(Net, self).__init__()

        # 输入通道、输出通道数、卷积核大小（一般长宽一致）
        # 其余的stride（步长）默认为1，padding（填充）为0
        # dilation（点积）一般不采用，bias一般包含在输出通道中
        self.conv1 = nn.Conv2d(1, 6, 3)
        
        # 考虑MNIST数据集和设计的神经网络结构，
        # 维度1x32x32 经conv1，维度变换为 6x(32-3+1)x(32-3+1)，
        # 在经过一个2x2的最大操作池化，维度变换为 6x(30/2)x(30/2)=1350
        self.fc1 = nn.Linear(1350, 10)

    def forward(self, x):
        """
        定义网络结构，也就是拟合函数，backward计算由nn.Module自动完成
        这个基本网络的设计是一个卷积层conv1，然后用relu激活，
        接一个最大操作池化层，接一个卷积层fc1
        输入是 batchsize x 1 x 32 x 32，即MNIST数据集的大小
        """
        
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        # x.size()[0]是batchsize，-1表示根据大小自适应
        # x.view相当于不改变张量元素个数的情况下将其维度重排布
        x = x.view(x.size()[0], -1)

        x = self.fc1(x)

        # 这个神经网络包含的可学习参数是一层conv1和一层fc1贡献的
        # 参数总量为6x(1+1x3x3) + (1350+1)x10 = 13570

        return x

# 查看网络的基本参数信息
def check_params(net: nn.Module, need_grad=True):
    # net.named_parameters() 字典数据结构，包含计算图各层信息
    for name, parameters in net.named_parameters():
        my_print('计算图参数', f"{name}: {parameters}")
    # 查看计算图各层的size大小
    for name, parameters in net.named_parameters():
        my_print('计算图大小', f'{name}: {parameters.size()}')
    # 查看计算图的梯度大小
    for name, parameters in net.named_parameters():
        my_print('计算图梯度信息', f"{name}: {parameters.grad}")

# 初始化网络参数
def init_weights(module: nn.Module):
    # 如果是全连接层，用全0初始化
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)
    # 如果是卷积层，用xavier正态分布初始化
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)

def criterion(out: torch.tensor, y: torch.tensor):
    loss = nn.MSELoss()
    return loss(out, y)

def main():

    net = Net()

    # apply函数递归遍历子模块，用其参数中的函数对所有子模块进行操作
    # net.apply(init_weights)

    # 测试输入和输出
    inp = torch.ones(12, 1, 32, 32)
    out = net(inp)

    # 假定标签
    y = torch.arange(0, 120).view(12, 10).float()

    # 根据预先定义好的loss表达式算出loss在当前样本的值
    loss = criterion(out, y)

    # 计算此时的梯度，（利用该梯度更新参数值）
    loss.backward()

    check_params(net)

    my_print('测试输出', loss)


if __name__ == '__main__':
    main()
