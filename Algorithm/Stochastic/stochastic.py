import numpy as np
import matplotlib.pyplot as plt

# 分析时间序列的趋势变化（均值、波动性等）
# 分析时间序列的自相关性和周期性
# 分析时间序列的分布特征

# 定义自己的输出格式
def my_print(description: str, x):
    if x is not None:
        print(f'---------------------------- {description} -----------------------------------')
        print(x)
        print('----------------------------- end ----------------------------------------')
    else:
        print(f'Start: ---------------------------- {description} -----------------------------------')
    print()

# 时间序列生成

# 随机过程：布朗运动的生成
# 包含四个参数：
# num_points 时间序列中的点数。
# step_size: 每一步的尺度因子，用于调整步长的大小。
# avg: 步长的平均值，默认为0。
# delta: 步长的标准差，默认为0.1。
def brown_time_series_gen(num_points, step_size, avg=0, delta=0.1):

    # 首先，创建一个长度为 num_points 的数组 time_series，所有元素初始值为0。
    time_series = np.zeros(num_points)

    # 将时间序列的第一个点设为平均值 avg。
    time_series[0] = avg
    for i in range(1, num_points):

        # 生成一个正态分布的随机数，其平均值为 avg，标准差为 delta，
        # 并乘以 step_size 以得到步长。
        step = np.random.normal(avg, delta) * step_size

        # 将这个步长加到上一个时间点的值上，得到当前时间点的值。
        time_series[i] = time_series[i-1] + step

    # 返回生成的时间序列数组
    return time_series

# 和布朗运动的生成方式差不多，不过带有上下界限制
# lb: 下界；ub：上界
def lb_brown_time_series_gen(num_points, 
    step_size, avg=0, delta=0.1, 
    lb = 0, ub = 100000
):
    time_series = np.zeros(num_points)
    time_series[0] = avg
    for i in range(1, num_points):
        step = np.random.normal(avg, delta) * step_size

        # 使用 np.clip 函数确保时间序列的值在lb和ub之间。
        time_series[i] = np.clip(time_series[i-1] + step, lb, ub)
    return time_series

# 时间序列里前后值没有关联关系，共有4个参数
# num_points: 时间序列中的点数。
# step_size: 步长的尺度因子，用于调整噪声的大小。
# avg: 噪声的平均值，默认为0。
# delta: 噪声的标准差，默认为0.1。
def noise_time_series_gen(num_points, step_size, avg=0, delta=0.1):

    # 创建一个长度为 num_points 的数组 time_series，初始时所有元素都是0。
    time_series = np.zeros(num_points)

    # 生成一个同样长度的噪声数组noise。这个噪声是由np.random.normal函数产生的，
    # 生成正态分布随机数数组，平均值为avg，标准差为delta，长度为num_points。
    # 将 noise 数组乘以 step_size，调整噪声的规模。
    noise = step_size * np.random.normal(avg, delta, num_points)

    # 将 time_series 数组和缩放后的 noise 数组相加，生成最终的时间序列。
    time_series = time_series + noise

    # 返回生成的有噪声的时间序列
    return time_series

# 时间序列的分析

# 计算并返回一个时间序列的全局平均值列表
# 这个函数可以用来分析时间序列数据的累积平均趋势
def global_mean_function(series):

    # 函数返回一个列表，其中的每个元素
    # 代表了原始序列中对应位置之前（包括该位置）所有元素的平均值。
    return [np.mean(series[:i+1]) for i in range(len(series))]

# 计算并返回一个时间序列的全局方差值列表
# 帮助识别数据随时间变化的离散程度
def global_variance_function(series):

    # 函数返回一个列表，其中的每个元素
    # 代表了原始序列中对应位置之前（包括该位置）所有元素的方差。
    return [np.var(series[:i+1]) for i in range(len(series))]

# 用于计算两个时间序列之间的累积协方差
# 协方差是衡量两个变量如何一起变化的统计度量。
# 随机变量有协方差矩阵，将时间序列的值看作样本，不同的时间序列之间也有协方差矩阵
def covariance_function(series1, series2):

    # np.cov(series1, series2) 生成一个 2x2 的协方差矩阵
    # 矩阵的对角线元素表示各自序列的方差
    # 非对角线元素表示两个序列之间的协方差，显然这是一个对角矩阵
    # np.cov(series1, series2)[0, 1] == np.cov(series1, series2)[1, 0]

    # 函数返回一个列表，其中每个元素
    # 是到当前索引为止的 series1 和 series2 之间的协方差。
    return [ np.cov(series1[:i+1], series2[:i+1])[0, 1] for i in range(len(series1)) ]

# 用于计算给定时间序列的时间渐进全局n阶矩（moment）
# 矩是统计学中用于描述数据分布特性的度量
# 更高阶的矩（如 n=3 和 n=4）分别关联于
# 数据的偏度（skewness，即分布的不对称性）
# 和峰度（kurtosis，即分布的尖峭或平坦程度）
def global_moment_function(series, n):
    moment = np.zeros(len(series))
    for i in range(len(series)):

        # 计算序列开始到当前索引i（包含索引 i）的所有元素的平均值 mean_i。
        mean_i = np.mean(series[:i+1])
        for j in range(i+1):
            moment[i] = moment[i] + np.power(series[j] - mean_i, n)

        # 完成内层循环后，将 moment[i] 的值除以 (i+1)，
        # 即到目前为止包含的元素个数，得到第 i 个位置的n阶矩。
        moment[i] = moment [i] / (i+1) 
    return moment

# 计算不同延迟下的自协方差，（基本不会用到）
def autocovariance_function(series, lag):
    return np.cov(series[:-lag], series[lag:])[0, 1]

# 计算并返回一个时间序列与其自身的自相关（autocorrelation）值。
# 自相关是一种度量信号与其自身在不同时间点的相似性的统计工具。

# 如果一个时间序列具有周期性，那么在特定的时间延迟，即
# 等于或是周期长度的整数倍下，自相关值会出现显著的峰值。

# 在零延迟（时间位移为0）下，自相关值总是最大的，
# 因为这时序列与其完全未移动的自身进行比较。
# 如果序列是纯随机的（白噪声），其自相关值将在零延迟之后迅速下降至接近零。
def global_autocorrelation_function(series):

    # 'full'模式下输出的长度是2n-1，n是原先时间序列的长度
    # 例：
    #      ------ （假设时间序列长度为6）
    # ------    ------ （在6*2-1个位置计算，）
    #      *********** （输出格式，每个位置对应相乘后累加得到输出值）
    return np.correlate(series, series, mode='full')

# 用于计算给定时间序列的功率谱密度（Power Spectral Density, PSD）
# PSD用于描述信号的频率内容和功率分布

# FFT（Fast Fourier Transform，简称FFT）是一种算法，
# 它能够高效地将一个时间序列（通常是一维的信号）从时域转换到频域。
# 在频域中，信号被表示为不同频率的成分的组合。

# 在FFT结果中，第一个元素（索引为0）代表直流分量，
# 即频率为0 Hz的分量，它表示时间序列的平均值或偏置。
# FFT结果中的中间元素（索引为N/2）代表Nyquist频率分量
# 即频率为采样率一半的分量。

# FFT结果中的前N/2+1个元素（对于奇数N，N/2向下取整）包含了唯一的频率信息。
# 剩余的元素是这些频率的复共轭，因此并不提供额外的唯一频率信息。

# FFT结果中的振幅较大的频率分量是识别时间序列周期性的重要线索。
# 振幅较大的频率分量是识别时间序列周期性的重要线索
def power_spectral_density(series):

    # 先算FFT，再平方除以时间
    return np.abs(np.fft.fft(series)) ** 2 / len(series)

# 计算给定时间序列的瞬时分布，即该序列的直方图分布。
# density表示是否用概率“密度”函数表示，
# 注意这个概率“密度”需要乘上bin的距离才能相加为1
def instantaneous_distribution(series, **kwargs):

    # bins='auto' 参数指示 np.histogram 自动选择箱的数量和范围
    hist, bins = np.histogram(series, bins='auto', **kwargs)

    # hist 是一个数组，包含每个箱中的数据点的密度（或数量）
    # bins 是一个数组，包含直方图箱的边界值。这些值定义了每个箱的范围。
    return hist, bins

# 用于计算给定时间序列中每个独特值的出现概率。
def transition_probability(series):

    # np.unique 函数用于找出序列中的所有独特值，并且计算每个独特值出现的次数。
    unique_values, counts = np.unique(series, return_counts=True)
    probabilities = counts / len(series)

    # 返回独特值和概率
    return unique_values, probabilities

# 画一条时间序列的前n个点
def plot_one_series(num_points, time_series):
    # plt.style.use('ggplot')
    plt.plot(time_series[:num_points])
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['font.family'] = 'Helvetica'
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 15
    plt.xlabel(r'Time Slot $\tau$', fontsize=18, fontfamily='Comic Sans MS', fontstyle='oblique', fontweight='normal')
    plt.ylabel('Value', fontsize=18, fontfamily='Times New Roman', fontstyle='oblique')
    plt.title('Time series', fontsize=18, fontfamily='Arial', fontstyle='italic', fontweight='light')
    # plt.grid(which='both', color='#0000FF', linewidth=10, linestyle='dashed')
    plt.show()

# 画两条时间序列的前n个点
def plot_two_series(num_points, time_series1, time_series2):
    plt.plot(time_series1[:num_points])
    plt.plot(time_series2[:num_points])
    plt.xlabel(r'Time Slot $\tau$', fontsize=18, fontfamily='Comic Sans MS', fontstyle='oblique', fontweight='normal')
    plt.ylabel('Value', fontsize=18, fontfamily='Times New Roman', fontstyle='oblique')
    plt.title('Time series and other series', fontsize=18, fontfamily='Arial', fontstyle='italic', fontweight='light')
    plt.show()

# 画至多5条时间序列的num_start到num_end的点
# 传入4个参数
# 第一.二个参数表示起始和终止的时间戳
# 第3个参数表示时间序列列表,最后一个参数表示label列表,应该和时间序列列表同等长度
def plot_multi_series(num_start, num_end, time_series: list, label_list: list):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(time_series)):
        # plt.plot(time_series[i], color=colors[i])
        plt.plot(time_series[i][num_start:num_end], label = label_list[i])
    plt.xlabel(r'Time Slot $\tau$', fontsize=18, fontfamily='Times New Roman', fontstyle='normal', fontweight='normal')
    plt.ylabel('Request Number', fontsize=18, fontfamily='Times New Roman', fontstyle='normal')
    plt.title('Different Edges Request Number', fontsize=18, fontfamily='Times New Roman', fontstyle='normal', fontweight='light')
    plt.legend(loc=2)
    plt.grid()
    plt.show()

def main():
    # default: avg = 1, delta = 0.1
    time_series = brown_time_series_gen(1000, 1)
    mean_series = global_mean_function(time_series)
    var_series = global_variance_function(time_series)
    cor_series = global_autocorrelation_function(time_series)[0:1000]
    moment_2_series = global_moment_function(time_series, 4)

    # a = np.ones(5) + noise_time_series_gen(5, 1, 1, 0.1)
    # b = np.ones(5) + noise_time_series_gen(5, 1, 1, 0.1)
    # my_print('covariance: ', covariance_function(a, b))
    # my_print('autorelation: ', global_autocorrelation_function(a))
    # my_print('power_spectral_density', power_spectral_density(a))
    # my_print('distribution', instantaneous_distribution(a, density=False))

    # plot_two_series(1000, var_series, moment_2_series)
    time_series_list = [time_series, mean_series, var_series, moment_2_series]
    # label_list = ['time_series', 'mean_series', 'var_series', 'moment_series']
    # plot_multi_series(1000, time_series_list, label_list)

    # 用具有上下界的布朗运动时间序列构造4个
    time_series1 = np.clip(lb_brown_time_series_gen(400, 100, 0, 0.1), 0, 1000)
    time_series2 = np.clip(lb_brown_time_series_gen(400, 100, 0, 0.1), 0, 1000)
    time_series3 = np.clip(lb_brown_time_series_gen(400, 100, 0, 0.1), 0, 1000)
    time_series4 = np.clip(lb_brown_time_series_gen(400, 100, 0, 0.1), 0, 1000)

    # 求和
    time_series = time_series1 + time_series2 + time_series3 + time_series4

    time_series_list = [time_series1, time_series2, time_series3, time_series4, time_series]
    label_list = ['Edge-1', 'Edge-2', 'Edge-3', 'Edge-4', 'Total Request']

    # 画出5个时间序列的趋势变化
    plot_multi_series(50, 400, time_series_list, label_list)
    plot_one_series(1000, var_series)

if __name__ == '__main__':
    main()

    


