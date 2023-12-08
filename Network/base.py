import torch
import numpy as np
print(torch.__version__)

# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

# torch.tensor和np.array是torch和numpy这两个库的基本数据结构，基本四则运算均被重载为针对元素的逐一操作。
def add_np(x: np.array, y: np.array):
    z = x + y
    return z

def add_torch(x: torch.tensor, y: torch.tensor):
    z = x + y
    return z

# list的+重载表示拼接，其余运算没有重载
def add_list(x: list, y: list):
    z = x + y
    return z

# torch和numpy的不同之处在于 1.torch可以转移到gpu做运算；2.torch支持自动求导
def move_tensor_to_device(data):
    # 检查设备是否具有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将张量移动到设备上
    data = data.to(device)

    # tensor.device 属性获取张量的位置信息
    return data, data.device

# 关于第一点：
# 以下代码展示torch.tensor可以和numpy.array互相运算，torch提供了这种重载，
# 同时gpu上的tensor也可以和cpu上的data进行运算，这个切换由torch自动完成
def test_gpu_tensor_with_cpu_nparray():
    np_array = np.ones(10)
    # 从numpy.array转化成tensor
    tensor_array = torch.from_numpy(np_array)
    # gpu上的tensor，cpu上的np.array
    move_tensor_to_device(tensor_array)
    my_print('test_gpu_tensor_with_cpu_nparray', tensor_array + np_array)

# 关于第二点：
# torch如何完成自动求导：首先明确求导是针对连续可微函数而言的
# torch是将导数保存在自变量上的，当因变量关于自变量的表达形式发生变化
# 也就是函数形式发生变化，这个值也就变化了，torch将该函数称为计算图。
# torch中要求因变量必须是标量，不能是多元向量。
# 以下代码展示了torch如何执行自动求导。
def test_autograd():
    my_print('test_autograd', None)
    # 创建一个需要求导的张量，并设置requires_grad=True
    x = torch.tensor(2.0, requires_grad=True)

    # 定义一个计算图（函数），先有函数再有导数
    y = x**2 + 3*x + 1

    # 使用backward()方法计算导数
    y.backward()

    # 访问导数值，导数值保留在自变量上
    gradient = x.grad

    my_print('Gradient:', gradient)

    # 修改函数定义
    y = x**3 + 2*x - 1

    # 再次使用backward()方法计算导数
    x.grad.zero_()  # 清零之前的导数值，如果不清零就是累加
    
    y.backward()
    my_print('Gradient (2nd backward):', x.grad)

def main():
    x, y = np.ones(10), np.ones(10)
    my_print('numpy add', add_np(x, y))

    x, y = torch.ones(10), torch.ones(10)
    my_print('tensor add', add_torch(x, y))

    x, y = list(range(10)), list(range(10))
    my_print('list add', add_list(x, y))

    test_gpu_tensor_with_cpu_nparray()

    test_autograd()
    
if __name__ == '__main__':
    main()