import torch
import numpy as np
print(torch.__version__)

def my_print(description: str, x):
    print(f'---------------------------- {description} -----------------------------------')
    print(x)
    print('----------------------------- end ----------------------------------------')

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

def main():
    x, y = np.ones(10), np.ones(10)
    my_print('numpy add', add_np(x, y))

    x, y = torch.ones(10), torch.ones(10)
    my_print('tensor add', add_torch(x, y))

    x, y = list(range(10)), list(range(10))
    my_print('list add', add_list(x, y))
    
if __name__ == '__main__':
    main()