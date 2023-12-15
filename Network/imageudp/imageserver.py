# 一个基于UDP的，支持图片传输保序性的，快速的，网络协议
# 服务器端实现
import socket

# 处理图像数据
import cv2

# 调用math.ceil()上取整函数
import math

# 调用sys.exit()函数
import sys

# 用来获取时间戳
import time

# python实现数据序列化（从而网络传输）
# 【传的是字节流，而字节流本身没有含义】
import pickle

# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

def buffer_to_image(write_file, image):
    pass

def recv_req():
    # '0.0.0.0'表示接受所有ip地址的消息
    host = '0.0.0.0'
    port = 6666

    # 关于这个值的说明：
    # 一般MTU（最大传输单元）在1500以下，超过的称为Jumbo frame
    # 相比于计算机的计算能力来说，网络传输的大小确实是有限制的
    # 这是因为网络的不稳定性，尤其是依赖电磁波传输的无线网

    # 然而服务器端可能会同时接收多个发送端的消息，因此要设置的大一些
    max_length = 6400

    buffer = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 对于客户端，其发送端口由操作系统分配，因此bind的过程是隐式的
    # 在服务器端，接收端口和地址的绑定要显示指出，
    # 当然，服务器获取客户端的sock之后也可以发送消息
    sock.bind((host, port))

    my_print('Server Init Finish! Waiting for requests!', None)

    cnt = 0
    while True:
        data, address = sock.recvfrom(max_length)
        frame_info = pickle.loads(data) 

        my_print(f'frame_info {cnt}', frame_info['n'])
        cnt += 1

def main():
    recv_req()

if __name__ == '__main__':
    main()