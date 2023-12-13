import cv2
print(cv2.__version__)

import os

# 对于图像类的数据，我们用python-opencv(cv2)进行处理，
# 注意cv2当中还能够处理视频类型数据，不过需要ffmpeg支持，
# cv2对于图像的一般性处理是很强的，但对于视频、神经网络等支持较差
# 因此仅调用cv2的图像一般处理功能，以及对于mp4格式视频的处理。

# 该文件展示了.mp4视频转化成神经网络张量输入的过程

def main():
    dataset_path = 'Dataset/tK-400/train_256/'
    pass

if __name__ == '__main__':
    main()
