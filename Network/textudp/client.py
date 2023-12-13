import socket
import cv2
print(cv2.__version__)

def send_data():
    host = '127.0.0.1'  # 服务器IP地址
    port = 6666  # 服务器端口

    # 创建套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # client_socket.connect((host, port))  # 连接服务器

    # 发送数据
    data = "要发送的数据"
    client_socket.sendto(data.encode('utf-8'), (host, port))

    client_socket.close() # 关闭连接

def main():
    send_data()

if __name__ == '__main__':
    main()