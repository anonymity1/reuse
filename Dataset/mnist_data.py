import torch
from torch.utils.data import Dataset, DataLoader
import struct
import numpy as np

import torchvision.transforms as transforms

# 自定义MNIST数据集类
# 定义一个数据集类，并构成一个可迭代对象，
# 使得训练或者测试时可以直接直接从中选用一个batch进行处理
# 即实现从硬盘到内存可迭代对象，需要两步走：1.读入内存；2.构建可迭代对象
# 分别对应：1.继承Dataset类，读入内存；2.用DataLoader读入，构建可迭代对象
# data_provider函数完成了这两个功能

# 继承Dataset类，关键是读取数据，以及重载两个函数（有点像一个表格的使用）：
# _read_data_函数(loc)从硬盘中读取数据
# 1.__len__()，获得dataset的长度；
# 2.__getitem__(index)，通过下标访问对应位置数据

def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

class MNISTDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        # MNIST的内容和标签分开存储
        self.images_file = images_file
        self.labels_file = labels_file

        # transform是对原始数据进行变换的函数，这是为了归一化的考虑
        self.transform = transform
        if self.transform is None:
            # transform是torchvision中的一个模块
            # Compose类的作用是将图像转换操作按顺序组合在一起，这些操作放在list里传给Compose作为参数
            # transforms.Compose(【...】)将列表中的函数封装成一个函数，
            # 这个函数接受图像类型数据（PIL.image打开的数据）或者ndarray类型数据
            self.transform = transforms.Compose([
                # transforms.ToTensor()这个函数可以参考其注释，非常全面，注意归一化到[0,1]区间了
                transforms.ToTensor(), 
                # transforms.Normalize([...],[...]) 对每个channel：out = (in - mean) / std
                # 第一个参数为图像通道样本总体平均值，第二个参数为图像通道样本总体标准差
                transforms.Normalize([0.5], [0.5]),
                # transforms提供的一个匿名函数的写法，但是ToTensor已经让原始图像转化为[C,H,W]了
                # transforms.Lambda(lambda x: x.unsqueeze(0))
            ])

        # 将硬盘上的数据读取到内存中
        self._read_data_(self.images_file, self.labels_file)

    def _read_data_(self, images_file, labels_file):
        
        # with function() as f, f是function()的返回值，不过这种写法不用销毁
        # 文件描述符是文件在内存中的数据结构，f.read(n)表示从该文件读取n个字节，返回n个字节
        with open(images_file, 'rb') as f:
            # magic变量用于存储数据集的魔术数字，用来标识数据集，rows和cols是每张图片的大小
            # struct.unpack中，'>IIII' 是一个格式字符串，指定了解析的方式。
            # 其中 > 表示使用大端字节序（big-endian）解析数据，
            # 后面的每个 I 表示解析一个 4 字节的无符号整数。
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            # np.fromfile()函数读取，注意它不具有解压功能
            # 注意MNIST像素值的保存是以uint8（一字节）的格式保存在硬盘上
            self.images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
            if self.transform is not None:
                self.images = [self.transform(image) for image in self.images]

        with open(labels_file, 'rb') as f:
            # 读取8个字节，两个数据
            magic, num_labels = struct.unpack('>II', f.read(8))
            self.labels = np.fromfile(f, dtype=np.uint8)

    def __len__(self):
        # 返回数据集长度
        return len(self.images)

    def __getitem__(self, index):
        # 在MNIST数据集中返回图像和对应标签
        image = self.images[index]
        label = self.labels[index]

        return image, label

# 在类外定义数据预处理函数，不用torchvision.transforms
# def transform(image):
#     image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
#     return image

# 根据三种不同用途定义，当然也可能有更多种用途
def data_provider(images_file, labels_file, flag, **kwargs):
    # 第一种用途是测试
    if flag == 'test':
        is_shuffle = False
        drop_last = True
        batch_size = kwargs.get('batch_size', 32)
    # 第二种用途是预测，batch里只有一个查询
    elif flag == 'pred':
        is_shuffle = False
        drop_last = False
        batch_size = 1
    # 第三种用途是训练
    else: # flag == 'train'
        is_shuffle = True
        drop_last = True
        batch_size = kwargs.get('batch_size', 32)
    
    # 首先完成数据集加载：从硬盘到内存
    dataset = MNISTDataset(images_file, labels_file)

    # 其次转化为可迭代对象，根据用途设定不同的batch，混合选项和最后batch丢弃选项
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_shuffle,
        drop_last=drop_last
    )

    # 返回内存种的数据集和转化后的可迭代对象
    return dataset, data_loader

def main():
    # 加载本地的MNIST训练集
    train_images_file = 'Dataset/MNIST/raw/train-images-idx3-ubyte'
    train_labels_file = 'Dataset/MNIST/raw/train-labels-idx1-ubyte'

    # 完成硬盘到内存的加载和转化为可迭代对象
    _, train_loader = data_provider(train_images_file, train_labels_file, flag='train')

    # 做一个测试，检验可迭代对象里的batch，输出batch序号
    cnt = 0
    for batch_data, batch_labels in train_loader:
        # 在这里进行模型训练或其他操作，这里打印能被100整除的batch序号的形状
        if cnt % 100 == 0:
            my_print(f'batch number: {cnt}', batch_data.shape)
        cnt = cnt + 1

if __name__ == '__main__':
    main()