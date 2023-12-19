# 使用线程池进行输出，线程之间无依赖关系
# 可以看到线程的输出打乱，说明线程之间无阻塞
import concurrent.futures

# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

def main():
    names = ["Alice", "Bob", "Charlie", "Dave"]

    # 存在上下文管理就可以用with，确保退出时可以释放资源，as表示别名
    # my_print()不是一个线程安全函数，因为里面的内容会被乱序执行，如果想要安全需要用None参数
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        
        # 提交任务给线程池处理，python的线程调度是由操作系统负责的，即线程数量和任务数量无对应关系
        futures = [executor.submit(my_print, f'from {name}: hello', name) for name in names]

        # 阻塞当前线程，等待其他线程返回，
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()