# LSTM模型：
# 参考Annotation/LSTM.png

# 输入： 序列长度*batchsize*特征维度

# 输出1（生成序列）：序列长度*batchsize*隐藏状态维度
# 输出2（序列最后一个时间步的隐藏状态）：1*batchsize*隐藏状态维度
# 输出3（序列最后一个时间步的对应生成）：1*batchsize*隐藏状态维度
# 注：输出2和输出3打包成一个元组进行输出

# LSTM参数计算：
# 总共4个门（输入门、遗忘门、细胞门、输出门）
# 每个门有两个权重矩阵和权重向量
# 第一层: (in + hid + 2) * hid
# 后续几层：（hid + hid + 2) * hid

import torch
import torch.nn as nn

def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

def main():
    # 创建LSTM层
    input_size = 10  # 输入特征的维度
    hidden_size = 1  # LSTM隐藏状态的维度
    num_layers = 10    # LSTM的层数

    lstm = nn.LSTM(input_size, hidden_size, num_layers)

    batch_size = 20

    # 创建输入数据
    input_seq = torch.randn(5, batch_size, input_size)  # 序列长度为5，批大小为1，特征维度为input_size
    hidden = (torch.zeros(num_layers, batch_size, hidden_size),  # 初始化隐藏状态
            torch.zeros(num_layers, batch_size, hidden_size))  # 初始化细胞状态

    # 前向传播
    output, hidden = lstm(input_seq, hidden)

    my_print('output shape', output.shape)
    my_print('hidden shape', hidden[0].shape)
    my_print('cell state shape', hidden[1].shape)

if __name__ == '__main__':
    main()
