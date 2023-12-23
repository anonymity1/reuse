# 根据指定要求生成符合分布的数据
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

# 生成具有平均值和标准差，并且用上下界截断有序的随机数组
# 每个数保留3位小数点，默认不排序(即axis=None)
def gen_ulb_avg(size: tuple, avg, delta, axis=None):

    # 生成上下界
    lb = avg - 2 * delta
    ub = avg + 2 * delta

    # 随机生成、截断、四舍五入
    data = np.clip(
        np.random.normal(avg, delta, size=size),
        lb, ub
    ).round(decimals=3)

    # 从小到大排序，在第axis维
    if isinstance(axis, int):
        data = np.sort(data, axis=axis)
    return data

def main():
    my_print('gen data', gen_ulb_avg((4,5), 10, 2, axis=1))

if __name__ == '__main__':
    main()