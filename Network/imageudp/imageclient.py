# 一个基于UDP的，支持图片传输保序性的，快速的，网络协议
# 客户端实现
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

# 实现
import select


# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

# 图像类型数据序列化
def image_to_buffer(image_file):
    # cv2读取本地图像数据
    image = cv2.imread(image_file)
    my_print('image info', image.shape)

    # 用jpg格式编码成一维np.uint8数组，无损编码大概降了6倍数据量
    # 90编码质量降了10倍数据量
    # 机器为中心的图像编码器（神经编码器）？
    retval, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    my_print('buffer info', buffer.shape)

    # 错误处理，如果编码发生错误，退出程序
    if retval == False:
        my_print('Error', 'An encoding error occurred. Exiting the program!')
        sys.exit()

    # 从编码后的np.uint8数组转化为字节序列(bytes)

    # 在Python中，bytes和str（或称为string）是两种不同的数据类型。
    # 分别用于表示二进制数据和文本数据。
    # bytes是一种不可变的字节序列类型，表示原始的二进制数据。
    # 它由8位字节组成，每个字节可以表示0-255范围内的整数。
    # bytes对象使用前缀b来表示，例如b'hello'。
    # 在Python3中，字符串字面值默认为str类型，而不是bytes类型。
    # 如果需要在字符串字面值前加上b前缀，可以将其表示为bytes类型。
    # 需要注意的是，bytes和str之间的转换需要进行编码和解码操作。
    # 可以使用encode()方法将str转换为bytes，使用decode()方法将bytes转换为str。
    # 编码是将字符串转换为字节序列的过程，而解码是将字节序列转换为字符串的过程。
    # 这些操作需要指定相应的字符编码方式，例如UTF-8、ASCII等。
    # 机器为中心的文字编码器？

    # 这一步没有编解码，只是转换格式
    # 即cv2先编码图像，再用python的函数转换成字节流，
    # 其实不做也可以，后面的pick.dumps()也实现了序列化
    # 需要用np.frombuffer解析
    buffer = buffer.tobytes()
    buffer_size = len(buffer)
    my_print('buffer info', buffer_size)

    # pickle.dumps(buffer)没有采用压缩，有点像tobytes()
    # 不过由于版本实现不同问题会有安全风险
    # buffer1 = pickle.dumps(buffer)
    # my_print('image1', len(buffer1))
    return buffer_size, buffer

# 上传的请求(request variant)中应当包含请求数据，请求类型，以及请求要求的完成时延
# req_file表示请求对应的原始数据
# req_type包含两部分：1.数据格式；2.数据目标
# SLO表示请求的规定完成截止期限
def upload_request(req_file, req_type='img-obj', SLO=30):
    # 根据UDP协议的规范，UDP数据报的最大长度是由IP层的最大传输单元 (MTU) 决定的。
    # 一般情况下，IPv4网络的MTU大小为1500字节，而IPv6网络的MTU大小为1280字节。
    # 然而，需要注意的是，这些MTU大小是指IP层的数据报大小，并不考虑UDP头部和IP头部的大小。
    # 在UDP协议中，每个UDP数据报都包含一个8字节的UDP头部，而IP头部的大小取决于所使用的IP版本和选项。
    # max_length = 65000
    max_length = 1000

    # 指定目标服务器IP地址和端口 
    host = '127.0.0.1'
    # host = '8.8.8.8'
    port = 6666

    # 系统调用，建立一个支持UDP协议的套接字
    # 第一个参数socket.AF_INET参数指定了套接字使用IPv4地址簇
    # 第二个参数socket.SOCK_DGRAM参数指定了套接字的类型，
    # 表示创建一个支持数据报传输（UDP）的套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 支持多种数据格式的处理【音频，视频，图像，文字】
    parts = req_type.split('-')
    if parts[0] == 'img':
        # 如果是图像数据，就调用image_to_buffer
        buffer_size, buffer = image_to_buffer(req_file)
    elif parts[0] == 'video':
        pass
    elif parts[0] == 'text':
        pass
    elif parts[0] == 'voice':
        pass
    else:
        pass

    # 根据单一数据报的长度定义数据报的数量
    num_of_packs = math.ceil(buffer_size / max_length)

    # UDP报头包含(src,dst)，因此在报头内部根据时间戳定位用户的请求
    # 定义数据报的顺序，每个数据报要有标号和时间戳
    # 第一个数据报需要添加SLO和服务信息，结尾数据报需要有标志

    # 获取当前时间戳
    timestamp = time.time()

    # 第0个数据报应该包含的信息：
    first_frame_info = {
        't': timestamp,
        # 根据req_file获得
        'n': num_of_packs,
        # 这两个信息依赖输入的req_type和SLO，第一个包的特有信息
        's': req_type,
        'l': SLO
    }
    my_print('first_frame_info', first_frame_info)
    first_frame_info = pickle.dumps(first_frame_info)
    my_print('first_frame_length', len(first_frame_info))
    # 发送第一帧数据
    # 这个方法没有返回值，但可能有异常，异常需要在服务端查看
    sock.sendto(first_frame_info, (host, port))
    
    # 浮点数序列化过程，需要先转为字符串再序列化，
    # python没有提供其直接序列化的工具
    # timestamp = str(timestamp).encode('utf-8')
    # my_print('timestamp', len(timestamp))
    left, right = 0, max_length
    for i in range(num_of_packs):
        # t,n,c是基本信息，
        # 但是除此之外需要让服务器可以确认接收已经结束，
        # 通过对比第一帧的'n'来获取
        frame_info = {
            't': timestamp,
            'n': i,
            'c': buffer[left:right]
        }
        left = right
        right += max_length
        frame_info = pickle.dumps(frame_info)
        sock.sendto(frame_info, (host, port))

    # 返回这两个值供接收函数调用
    return sock, SLO

# 当用户需要得到推断结果时，需要非阻塞式接收结果
# 对于召测型数据，可能是云端需要推断结果，就不需要客户端再接收
def unblock_recv(sock, SLO):

    # 设置成非阻塞模式，准备接收来自服务器的推断结果
    # 注意：阻塞和非阻塞是指在进行系统调用时的行为方式
    # 即内核空间的状态
    sock.setblocking(False)

    # 注意这里如果是本地IP，select会直接返回信息打印
    # 因为本地会的防火墙会直接拒绝监听
    readable, _, _ = select.select([sock], [], [], SLO)
    my_print('readable', readable)
    if readable:
        response, server_address = sock.recvfrom(4096)
        my_print('reponse', f"recv: {response.decode()}")
    else: 
        my_print('Error', 'SLO Broken')

    # 这段代码尽管对于内核是非阻塞模式，
    # 但是用户空间会一直执行这段代码，占据CPU资源
    # 如果用本地回换链路，可以用这段代码作为select的替代
    # while True:
    #     try:
    #         # 尝试接收服务器的响应消息
    #         response, server_address = sock.recvfrom(4096)
    #         print(f"Received response from server: {response.decode()}")
    #         break  # 接收到响应后跳出循环
    #     except socket.error as e:
    #         # 如果没有接收到响应，处理其他逻辑
    #         pass

    return readable

def main():
    # 请求文件
    req_file = 'Network/Testdata/0.png'
    # 根据req_file, req_variant向所属边缘上传推断请求
    sock, SLO = upload_request(req_file=req_file, req_type='img-obj', SLO=30)
    # （可选）非阻塞式等待响应
    response = unblock_recv(sock, SLO)

    my_print('End', '')

if __name__ == '__main__':
    main()