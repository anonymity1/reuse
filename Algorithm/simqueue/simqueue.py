# 一台设备同一时间只能执行一个任务，采用FIFO的方式执行任务。
# 输入：一组任务的到达时间，每个任务的执行时间。
# 输出三个数组：每个任务的排队时间、每个任务的结束时间、设备在每个时间段内执行的任务序号，
# 如果没有执行输出空转。用python代码解决该问题。
import numpy as np

# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

# n个任务到达，输入每个任务的到达时间和执行时间
# 假设输入的任务是排好队的，机器同一时间只能执行一个任务
def sim_queue(arrival_times, execution_times):
    n = len(arrival_times)
    queue_times = [0] * n
    end_times = [0] * n
    execution_order = []

    current_time = 0

    # 模拟每个任务的执行，
    # 因为已知每个任务的到达时间和执行时间，和执行方式（FIFO）
    # 不需要采用模拟每个时间戳的方式
    for i in range(n):

        # 计算该任务的排队时间
        queue_times[i] = max(0, current_time - arrival_times[i])

        # 将设备执行时间推进到当前任务的到达时间
        current_time = max(current_time, arrival_times[i])

        # 执行当前任务，更新时间戳
        end_time = current_time + execution_times[i]

        end_times[i] = end_time

        # 执行完成
        execution_order.append(i)

        # 更新当前时间
        current_time = end_time

    # 任务的到达时间、每个任务的实际开始时间、任务的结束时间，
    execution_times_dict = {
        i: {
            'arrival_time': arrival_times[i], 
            'queue_time': queue_times[i],
            'start_time': end_times[i] - execution_times[i],
            'end_time': end_times[i]
        } for i in range(n)
    }

    # 返回每个任务的排队时间、结束时间、执行顺序、以及每个任务的到达时间、实际开始时间
    return queue_times, end_times, execution_order, execution_times_dict

def main():
    # 示例输入
    arrival_times = [0, 2, 4, 13]
    execution_times = [7, 4, 1, 3]

    # 执行任务
    queue_times, end_times, execution_order, execution_times_order = sim_queue(arrival_times, execution_times)

    # 打印结果
    my_print("每个任务的排队时间:", queue_times)  
    my_print("每个任务的结束时间:", end_times)
    my_print("设备在每个时间段内执行的任务序号:", execution_order)
    for key, value in execution_times_order.items():
        my_print(f'第{key}个任务:', value)

if __name__ == '__main__':
    main()