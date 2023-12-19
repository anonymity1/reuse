# 这是一个精心设计的异步场景，包含三个异步线程
# 输入IO、加法操作和乘法操作三个异步线程

# 实际应用中异步的产生原因是消费者和生产者的工作速度不一致
# 如果是同步程序会发生“停等”，等待是有必要的，但是停应当避免
# 同步程序如果避免“停”需要大量的if-else判断和时间监控
# 异步程序利用操作系统提供的线程工具和队列通信工具，
# 它对于挂起和执行的判断由操作系统对队列监控完成

# 构造线程
import threading

# 队列通信需要
import queue

# 退出程序
import sys

# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

# 用于接收标准输入的线程函数
# 并将输入数字传入加法队列中
def input_thread(q: queue.Queue):
    # 获取用户输入
    number = input("请输入一个数字：")
    
    # 判断是不是数字
    try:
        int_number = int(number)
    except ValueError:
        my_print('ValueError', 'Exit!')
        sys.exit()

    # 将输入的数字放入队列
    q.put(int_number)

# 用于处理输入的线程函数，将加法队列中的前n个数进行累加
# 输出n个数到乘法队列，每个数分别进行1到n次方
def thread_add(add_q: queue.Queue, multi_q: queue.Queue, n=2):
    while True:
        # 从队列中获取数字
        num, res = n, 0
        if add_q.qsize() >= num:
            for _ in range(num):
                res = res + add_q.get()

            for i in range(1, num + 1):
                multi_q.put(res ** i)
            my_print("输入的数字的和为:", res)

# 乘法线程，对乘法队列中前n个数进行累乘
# 输出这个结果为最终目标
# 接收传入的乘法队列作为输入
def thread_multi(multi_q: queue.Queue, n=3):
    while True:
        # 从队列multi_q中获取数字
        num, res = n, 1
        if multi_q.qsize() >= num:
            for _ in range(num):
                res = res * multi_q.get()
            my_print("输入的数字的乘积为:", res)

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

def main():
    # 创建一个队列用于线程间通信
    add_q= queue.Queue()
    multi_q = queue.Queue()

    # 启动两个子线程，等待队列中的消息，没有消息则挂起
    thread1 = start_thread(thread_add, add_q, multi_q)
    thread2 = start_thread(thread_multi, multi_q)

    # 启动接收输入的线程
    while True:
        input_thread(add_q)

    # 等待处理线程结束，这种情景只需各做各的即可
    # thread1.join()
    # thread2.join()

if __name__ == '__main__':
    main()