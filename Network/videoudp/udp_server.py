import cv2
import socket
import pickle
import numpy as np

host = "127.0.0.1"
port = 6666
max_length = 65540

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

frame_info = None
buffer = None
frame = None

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.moveWindow("Video", 100, 100)  # 设置窗口在屏幕上的位置

print("-> waiting for connection")

while True:
    data, address = sock.recvfrom(max_length)
    if len(data) < 100:
        frame_info = pickle.loads(data)
        
        if frame_info:
            nums_of_packs = frame_info["packs"]
            
            for i in range(nums_of_packs):
                data, address = sock.recvfrom(max_length)
                if i == 0:
                    buffer = data
                else:
                    buffer += data

            frame = np.frombuffer(buffer, dtype=np.uint8)
            frame = frame.reshape(frame.shape[0], 1)

            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)
            
            if frame is not None and type(frame) == np.ndarray:
                cv2.imshow("Stream", frame)
                cv2.waitKey(1)
                if cv2.waitKey(1) == 27:
                    break

cv2.destroyAllWindows()
                
print("goodbye")