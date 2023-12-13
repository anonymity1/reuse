import socket

def start_server():
    host = '127.0.0.1'  # 监听所有可用的网络接口
    port = 6666  # 服务器端口

    # 创建套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((host, port))  # 绑定地址和端口

    print("服务器已启动，等待连接...")

    while True:
        data, addr = server_socket.recvfrom(1024)  # 接收数据
        print("接收到的数据：", data.decode('utf-8'))

def main():
    start_server()

if __name__ == '__main__':
    main()