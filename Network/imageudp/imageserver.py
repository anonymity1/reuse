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

# numpy数组和字节流转换
import numpy as np

# 队列通信需要
import queue

# 构造线程
import threading

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

# 从字节流解码出原始图像数据
def buffer_to_image(buffer: bytes):

    # 从字节流转为numpy数组
    tmp = np.frombuffer(buffer, np.uint8)

    # 解码时会根据图片格式自动选择解码器
    # 第二个参数可选，如果图像本身是灰色的，那么它的第三通道值会一样
    image = cv2.imdecode(tmp, cv2.IMREAD_COLOR)

    return image

# 创建并启动子线程，包含三个参数，
# 第一个是函数名，第二个是参数列表，第三个是关键字列表
def start_thread(thread_func, *args, **kwargs):
    thread = threading.Thread(target=thread_func, args=args, kwargs=kwargs)
    # 设置为守护进程，确保主线程销毁子线程也销毁
    thread.daemon = True
    # 启动线程
    thread.start()

    # 返回线程名称
    return thread

# 第一个线程：从socket套接字接收消息并压入data_q队列
def thread_recv(sock: socket.socket, max_length: int, data_q: queue.Queue()):
    data, address = sock.recvfrom(max_length)
    data_q.put((data, address))

# 第二个线程：来自多用户多请求的分片数据报处理
def thread_data(data_q: queue.Queue(), request_q: queue.Queue()):
    # 数据报缓冲区解决有多个用户请求同时到达的问题
    # 需要定期清除缓冲区内容
    data_buffer = {}
    while True:
        # TODO: 用元组和ID映射的形式构造缓存字典，降开销
        if data_q.qsize() >= 1:
            data, address = data_q.get()
            frame_info = pickle.loads(data) 
            
            # 用用户地址和时间戳标志唯一报文
            info1 = (address, frame_info['t'])

            # 第一个数据报，或者第一个数据报丢失导致异常
            if info1 not in data_buffer:
            # if address not in address_buffer.keys():
                
                # service type这个类型应在第一个数据报中出现
                # 如果没出现，说明该请求的第一个数据报丢包
                # 将request解析失败结果压入request_q队列中
                if 's' not in frame_info:
                    data_buffer[info1] = ['failed']
                    request_q.put((info1, 'failed', None))

                # 如果出现service type信息，开始处理该数据报
                else:
                    # 存在service type信息，n和l也一定存在
                    try:
                        assert 'n' in frame_info, "key word n not in frame info"
                        assert 'l' in frame_info, "key word l not in frame info"
                    except AssertionError:
                        data_buffer[info1] = ['failed']
                        request_q.put((info1, 'failed', None))
                    cnt = 0 # 用cnt标志处理到的位置

                    # 第一位标志处理状态，第二位标志处理位置
                    # 第三、四、五位标志request的基本信息
                    # 第六位是字节流信息
                    data_buffer[info1] = [
                        'success',
                        cnt,
                        frame_info['n'], 
                        frame_info['s'], 
                        frame_info['l'], 
                        b''                   
                    ]

            # data_buffer中有该请求信息，即非第一个数据报情况
            else:
                # 如果是第一个数据报丢失的错误信息，直接把该数据报丢弃
                if data_buffer[info1][0] == 'failed':
                    continue

                # 另外一种是正常情况，数据报不丢弃，开始解码
                else:
                    # 确认数据报的位置，这说明数据报是按顺序正常解析的
                    if frame_info['n'] == data_buffer[info1][1]:

                        # 合并字节流
                        data_buffer[info1][5] += frame_info['c']
                        # 数据报处理位置++
                        data_buffer[info1][1] += 1

                        # 判断解析到的位置是不是已经结束
                        if data_buffer[info1][1] == data_buffer[info1][2]:
                            request_q.put((info1, 'success', data_buffer[info1]))
                        
                        # 测试一下是否正常
                        # my_print('data_buffer', data_buffer[info1][1])
                            
                    # 如果位置不正常，说明数据报解析不正常
                    else:
                        data_buffer[info1][0] = 'failed'
                        request_q.put((info1, 'failed', None))

# 第三个线程：将request队列中的请求根据需求解码
# 并转到对应的推断模型进行处理
def thread_request(request_q: queue.Queue(), sock:socket.socket):
    request_buffer = {}
    while True:
        if request_q.qsize() >= 1:
            info1, is_success, requeset_info = request_q.get()

            # 首先检验request是否成功
            if is_success == 'failed':
                # TODO: 将结果返回用户或直接处理

                # 这是将结果返回用户的示例
                # 因为是一段话，不用切割报文
                response = f'request send at {info1[1]} {is_success}!'
                sock.sendto(response.encode(), info1[0])

                # 出现错误跳过之后的分支判断，取下一个请求
                continue

            if requeset_info[3].split('-')[0] == 'img':
                image = buffer_to_image(requeset_info[5])

                response = f'request send at {info1[1]} {is_success}!'
                sock.sendto(response.encode(), info1[0])

                # 测试一下是否解码成功
                # cv2.imwrite('222.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            elif requeset_info[3].split('-')[0] == 'video':
                pass
            elif requeset_info[3].split('-')[0] == 'text':
                pass
            elif requeset_info[3].split('-')[0] == 'voice':
                pass         
            else:
                my_print('Error', 'format error')   

            # 获得request-variant
            SLO = requeset_info[4]
            infer = requeset_info[3].split('-')[1]

def start_server():
    # '0.0.0.0'表示接受所有ip地址的消息
    host = '0.0.0.0'
    port = 6666

    # 关于这个值的说明：
    # 一般MTU（最大传输单元）在1500以下，超过的称为Jumbo frame
    # 相比于计算机的计算能力来说，网络传输的大小确实是有限制的
    # 这是因为网络的不稳定性，尤其是依赖电磁波传输的无线网

    # 然而服务器端可能会同时接收多个发送端的消息，因此要设置的大一些
    max_length = 6400

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 对于客户端，其发送端口由操作系统分配，因此bind的过程是隐式的
    # 在服务器端，接收端口和地址的绑定要显示指出，
    # 当然，服务器获取客户端的sock之后也可以发送消息
    sock.bind((host, port))

    my_print('Server Init Finish! Waiting for requests!', None)

    # 三线程信息交互需要的队列 
    data_q, request_q = queue.Queue(), queue.Queue()

    # 开启数据报处理线程
    start_thread(thread_data, data_q, request_q)

    # 开启请求处理线程
    start_thread(thread_request, request_q, sock)

    # 接收消息的主线程
    while True:
        # socket套接字接收消息压入数据报队列线程
        thread_recv(sock, max_length, data_q)

def main():
    start_server()

if __name__ == '__main__':
    main()