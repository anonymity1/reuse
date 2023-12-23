import gurobipy as gp
from gurobipy import GRB

# 优化问题三要素：目标（setObjective）、约束（addConstr）、待求解变量（addVar）

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

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
    return data# 解析参数，在使用gurobi求解的时候会用到

class Params():
    def __init__(self, 
        req=0, loss=0, model_mem=0, edge_mem=0, zeta=0, net_u=0, last_x=0, xi=0, net_d=0, com=0, deploy=0,
        app_num=3, edge_num=4, inference_num=5, 
        loss_avg=0.3, loss_delta=0.1, req_avg = 6, req_delta = 1, 
        model_mem_avg = 1, model_mem_delta = 0.1,  edge_mem_avg = 5, edge_mem_delta = 0.1,
        zeta_avg = 1, zeta_delta = 0.1, net_u_avg = 8, net_u_delta = 1,
        xi_avg = 1, xi_delta = 0.1, net_d_avg = 8, net_d_delta = 1,
        com_avg = 1, com_delta = 0.1, deploy_avg = 2, deploy_delta = 0.1,
        tau = 150
    ):
        # 周期性推断负载重分布问题需要的参数
        # app_num, edge_num, inference_num 对应应用数量、边缘数量和推断模型数量
        # i，j，k描述了问题的规模
        self.app_num = app_num
        self.edge_num = edge_num
        self.inference_num = inference_num

        # 推断请求，每个边缘，每个应用在一定时间内的请求数量
        self.req = np.ones((self.app_num, self.edge_num))

        # 推断loss，每个应用，每种模型的loss值
        self.loss = np.ones((self.app_num, self.inference_num))

        # 1.描述内存约束
        # 每个应用的每种模型的内存消耗
        self.model_mem = np.ones((self.app_num, self.inference_num))

        # 每个边缘的内存
        self.edge_mem = np.ones((self.edge_num))

        # 2.描述网络带宽约束
        # 每种应用每类推断模型的网络传输大小
        self.xi = np.ones((self.app_num, self.inference_num))

        # 每类推断应用请求的网络传输大小
        self.zeta = np.ones((self.app_num))

        # 每个边缘的上行带宽，上行带宽负责将请求上传
        self.net_u = 10 * np.ones((self.edge_num))

        # 每个边缘的下行带宽，下行带宽负责下载模型和请求
        self.net_d = 10 * np.ones((self.edge_num))

        # 上一次模型部署的选择
        self.last_x = np.zeros((self.app_num, self.inference_num, self.edge_num))

        # 3.描述计算约束
        # 每种应用每个模型在每个边缘上的部署时间
        self.deploy = np.zeros((self.app_num, self.inference_num, self.edge_num))

        # 每种应用每个模型在每个边缘上的计算时间
        self.com = np.zeros((self.app_num, self.inference_num, self.edge_num))

        self.loss_avg = loss_avg
        self.loss_delta = loss_delta
        self.loss_lb = loss_avg - loss_delta
        self.loss_ub = loss_avg + loss_delta

        self.req_avg = req_avg 
        self.req_delta = req_delta
        self.req_lb = req_avg - req_delta
        self.req_ub = req_avg + req_delta

        self.model_mem_avg = model_mem_avg
        self.model_mem_delta = model_mem_delta
        self.model_mem_lb = model_mem_avg - model_mem_delta
        self.model_mem_ub = model_mem_avg + model_mem_delta

        self.edge_mem_avg = edge_mem_avg
        self.edge_mem_delta = edge_mem_delta
        self.edge_mem_lb = edge_mem_avg - edge_mem_delta
        self.edge_mem_ub = edge_mem_avg + edge_mem_delta

        self.zeta_avg = zeta_avg
        self.zeta_delta = zeta_delta
        self.zeta_ub = zeta_avg + zeta_delta
        self.zeta_lb = zeta_avg - zeta_delta

        self.net_u_avg = net_u_avg
        self.net_u_delta = net_u_delta
        self.net_u_ub = net_u_avg + net_u_delta
        self.net_u_lb = net_u_avg - net_u_delta

        self.xi_avg = xi_avg
        self.xi_delta = xi_delta
        self.xi_ub = xi_avg + xi_delta
        self.xi_lb = xi_avg - xi_delta

        self.net_d_avg = net_d_avg
        self.net_d_delta = net_d_delta
        self.net_d_ub = net_d_avg + net_d_delta
        self.net_d_lb = net_d_avg - net_d_delta

        self.com_avg = com_avg
        self.com_delta = com_delta
        self.com_ub = com_avg + com_delta
        self.com_lb = com_avg - com_delta
        
        self.deploy_avg = deploy_avg
        self.deploy_delta = deploy_delta
        self.deploy_ub = deploy_avg + deploy_delta
        self.deploy_lb = deploy_avg - deploy_delta

        self.tau = tau

    def set_random(self):

        self.loss = gen_ulb_avg((self.app_num, self.inference_num), self.loss_avg, self.loss_delta, axis=1)

        self.req = np.round(
            np.clip(np.random.normal(self.req_avg, self.req_delta, size=(self.app_num, self.edge_num)), self.req_lb, self.req_ub)
        )

        self.model_mem = np.clip(
            np.random.normal(self.model_mem_avg, self.model_mem_delta, size=(self.app_num, self.inference_num)), 
            self.model_mem_lb, self.model_mem_ub
        ).round(decimals=2)

        self.model_mem = np.sort(self.model_mem, axis=1)[:, ::-1]

        self.edge_mem = np.clip(
            np.random.normal(self.edge_mem_avg, self.edge_mem_delta, size=(self.edge_num)), 
            self.edge_mem_lb, self.edge_mem_ub
        ).round(decimals=2)
        
        self.zeta = np.clip(
            np.random.normal(self.zeta_avg, self.zeta_delta, size=(self.app_num)),
            self.zeta_lb, self.zeta_ub
        ).round(decimals=2)
        
        self.net_u = np.clip(
            np.random.normal(self.net_u_avg, self.net_u_delta, size=(self.edge_num)),
            self.net_u_lb, self.net_u_ub
        ).round(decimals=2)
        
        self.xi = np.clip(
            np.random.normal(self.xi_avg, self.xi_delta, size=(self.app_num, self.inference_num)),
            self.xi_lb, self.xi_ub
        ).round(decimals=2)

        self.xi = np.sort(self.xi, axis=1)[:, ::-1]
        
        self.net_d = np.clip(
            np.random.normal(self.net_d_avg, self.net_d_delta, size=(self.edge_num)),
            self.net_d_lb, self.net_d_ub
        ).round(decimals=2)

        self.com = np.clip(
            np.random.normal(self.com_avg, self.com_delta, size=(self.app_num, self.inference_num, self.edge_num)),
            self.com_lb, self.com_ub
        ).round(decimals=2)

        self.com = np.sort(self.com, axis=1)[:, ::-1, :]

        self.deploy = np.clip(
            np.random.normal(self.deploy_avg, self.deploy_delta, size=(self.app_num, self.inference_num, self.edge_num)),
            self.deploy_lb, self.deploy_ub
        ).round(decimals=2)

        self.deploy = np.sort(self.deploy, axis=1)[:, ::-1, :]

        my_print('Loss Data [app_num, inference_num]', self.loss)
        my_print('Request Data [app_num, edge_num]', self.req)
        my_print('Model Mem [app_num, inference_num]', self.model_mem)
        my_print('Edge Mem [edge_num]', self.edge_mem)
        my_print('upload network [edge_num]', self.net_u)
        my_print('download network [edge_num]', self.net_d)
        my_print('app request size (zeta) [app_num]', self.zeta)
        my_print('Model transmission size (xi) [app_num, inference_num]', self.xi)
        my_print('Deploy latency [app_num, inference_num, edge_num]', self.deploy)
        my_print('Computing latency [app_num, inference_num, edge_num]', self.com)

    # 展示不同种类应用模型的推断精度、每个边缘的请求、模型网络传输size和模型内存占用四张图
    def show_para(self):
        app_label, infer_label, edge_label = [], [], []
        for i in range(self.app_num):
            app_label.append(f'App-{i}')
        for j in range(self.inference_num):
            infer_label.append(f'Infer-{j}')
        for i in range(self.edge_num):
            edge_label.append(f'Edge-{i}')

        # fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        im, _ = heatmap(self.loss, app_label, infer_label, ax=ax, cmap='coolwarm', cbarlabel='Loss')
        annotate_heatmap(im)

        im, _ = heatmap(self.req, app_label, edge_label, ax=ax2, cmap='coolwarm', cbarlabel='Request Number')
        annotate_heatmap(im)

        im, _ = heatmap(self.xi, app_label, infer_label, ax=ax3, cmap='coolwarm', cbarlabel='Model Transmission')
        annotate_heatmap(im)

        im, _ = heatmap(self.model_mem, app_label, infer_label, ax=ax4, cmap='coolwarm', cbarlabel='Model Memory')
        annotate_heatmap(im)

        fig.tight_layout()
        plt.show()

def heatmap(data, row_labels, col_labels, title=None, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    label_font_dict = {
        'fontsize': 15,
        'fontfamily': 'Times New Roman',
        'fontweight': 'normal'
    }

    title_fond_dict = {
        'fontsize': 18,
        'fontfamily': 'Times New Roman',
        'fontweight': 'normal'
    }

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontdict=label_font_dict)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontdict=label_font_dict)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    if title != None:
        ax.set_title(title, fontsize=18, fontfamily='Times New Roman')

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# 生成每种应用的x，y，b分布图
def show_xyb(args, x, y, b, app=0):
    fig, ((ax, ax2, ax3)) = plt.subplots(1, 3, figsize=(12, 3))

    app_label, infer_label, edge_label = [], [], []
    for i in range(args.app_num):
        app_label.append(f'App-{i}')
    for j in range(args.inference_num):
        infer_label.append(f'Infer-{j}')
    for i in range(args.edge_num):
        edge_label.append(f'Edge-{i}')

    im, _ = heatmap(x[app], infer_label, edge_label, ax=ax, cmap='coolwarm', cbarlabel='is_deploy')
    annotate_heatmap(im)

    im, _ = heatmap(y[app], edge_label, edge_label, ax=ax2, cmap='coolwarm', cbarlabel='Workload Transition')
    annotate_heatmap(im)

    im, _ = heatmap(b[app], infer_label, edge_label, ax=ax3, cmap='coolwarm', cbarlabel='Model Throughput')
    annotate_heatmap(im)

    fig.tight_layout()
    plt.show()


# 根据参数求解优化问题
# 输入变量是参数，输出是待求解变量和最优值
def prob(args):

    # 创建模型
    model = gp.Model()

    # 创建变量

    # 负载转移变量y, uint
    result_y = {}
    for i in range(0, args.app_num):
        for k in range(0, args.edge_num):
            for k_ in range(0, args.edge_num):
                # name只是注释
                name = 'result_y({},{},{})'.format(i, k, k_)
                # model.addVar()函数用来添加变量，用字典存储变量，lb，ub用来描述上下限
                result_y[(i, k, k_)] = model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=name)

    # 模型部署指示变量x，01，
    result_x = {}
    for i in range(0, args.app_num):
        for j in range(0, args.inference_num):
            for k in range(0, args.edge_num):
                # name只是注释
                name = 'result_x({},{},{})'.format(i, j, k)
                # model.addVar()函数用来添加变量，用字典存储变量
                result_x[(i, j, k)] = model.addVar(lb=0, ub=1, vtype=gp.GRB.INTEGER, name=name)

    # 模型承担负载变量b, uint
    result_b = {}
    for i in range(0, args.app_num):
        for j in range(0, args.inference_num):
            for k in range(0, args.edge_num):
                # name只是注释
                name = 'result_b({},{},{})'.format(i, j, k)
                # model.addVar()函数用来添加变量，用字典存储变量，lb，ub用来描述上下限
                # addVar()返回变量，用变量.x访问具体值
                result_b[(i, j, k)] = model.addVar(lb=0, vtype=gp.GRB.INTEGER, name=name)

    # 设置目标函数
    # 添加线性表达式
    objective = gp.LinExpr()
    for i in range(args.app_num):
        for j in range(args.inference_num):
            for k in range(args.edge_num):
                # addTerms给线性表达式添加项，第一个是系数，第二个是变量
                objective.addTerms(args.loss[i, j], result_b[(i, j, k)])
    # setObjective给求解问题model设置目标以及求最大还是求最小
    model.setObjective(objective, GRB.MINIMIZE)

    # 添加约束条件1：设备k向其他设备转移的负载迁移和等于设备本身产生的负载r_ik
    for k in range(args.edge_num):
        for i in range(args.app_num):
            expr = gp.LinExpr()
            for k_ in range(args.edge_num):
                expr.addTerms(1.0, result_y[(i, k, k_)])
            # addConstr添加约束
            model.addConstr(expr == args.req[i, k], name="App {}'s workload at edge {} limit".format(i, k))

    # 添加约束条件2：y,x,b三者之间关系，转移到设备k的负载应当被不同模型执行完毕
    for k in range(args.edge_num):
        for i in range(args.app_num):
            expr1 = gp.LinExpr()
            expr2 = gp.QuadExpr()
            for k_ in range(args.edge_num):
                expr1.addTerms(1.0, result_y[(i, k_, k)])
            for j in range(args.inference_num):
                expr2.addTerms(1.0, result_x[(i, j, k)], result_b[(i, j, k)])
            model.addConstr(expr1 == expr2, name="App {}'s workload at edge {} need to be done".format(i, k))

    # 添加约束条件3：x，b两者关系，先部署模型才能处理负载
    for k in range(args.edge_num):
        for i in range(args.app_num):
            for j in range(args.inference_num):
                model.addConstr(result_b[(i, j, k)] >= result_x[(i, j, k)], name=f"var_x_b: {i}, {j}, {k}")
                model.addConstr(result_b[(i, j, k)] == result_x[(i, j, k)] * result_b[(i, j, k)], name=f"var2_x_b: {i}, {j}, {k}")

    # 添加约束条件4：设备k内存约束
    for k in range(args.edge_num):
        expr = gp.LinExpr()
        for i in range(args.app_num):
            for j in range(args.inference_num):
                expr.addTerms(args.model_mem[i, j], result_x[(i, j, k)])
        model.addConstr(expr <= args.edge_mem[k], name=f"Memory Limit: {k}")

    # 添加约束条件5：设备间上行带宽约束
    for k in range(args.edge_num):
        expr = gp.LinExpr()
        for i in range(args.app_num):
            for k_ in range(args.edge_num):
                if k_ != k:
                    # 本地设备不会发生请求上传
                    expr.addTerms(args.zeta[i], result_y[(i, k, k_)])
        model.addConstr(expr <= args.net_u[k], name=f"Upload Memory Limit: {k}")

    # 添加约束条件6：设备间下行带宽约束
    for k in range(args.edge_num):
        expr = gp.LinExpr()
        for i in range(args.app_num):
            for k_ in range(args.edge_num):
                if k_ != k:
                    # 第一个部分是请求下载
                    expr.addTerms(args.zeta[i], result_y[(i, k_, k)])
        for i in range(args.app_num):
            for j in range(args.inference_num):
                if int(args.last_x[i, j, k]) == 0:
                    # 第二个部分是模型下载
                    expr.addTerms(args.xi[i, j], result_x[(i, j, k)])
        model.addConstr(expr <= args.net_d[k], name=f"Download Memory Limit: {k}")

    # 添加约束条件7：整体计算延迟限制
    # 每个设备上的每类应用，在一个负载重分布周期内，应该完成模型部署和推断请求的执行
    for k in range(args.edge_num):
        expr = gp.LinExpr()
        for i in range(args.app_num):
            for j in range(args.inference_num):
                expr.addTerms(args.com[i, j, k], result_b[(i, j, k)])
                if args.last_x[i, j, k] == 0:
                    expr.addTerms(args.deploy[i, j, k], result_x[(i, j, k)])
        model.addConstr(expr <= args.tau)

    # optimize()函数求解优化问题
    model.optimize()

    x = np.zeros((args.app_num, args.inference_num, args.edge_num))
    b = np.zeros((args.app_num, args.inference_num, args.edge_num))
    y = np.zeros((args.app_num, args.edge_num, args.edge_num))
    # 打印结果
    if model.status == GRB.OPTIMAL:
        my_print('Optimal objective value', model.objVal)
        for i in range(0, args.app_num):
            for j in range(0, args.inference_num):
                for k in range(0, args.edge_num):
                    # name = 'result_x({},{},{})'.format(i, j, k)
                    # print(f'{name}={result_x[(i, j, k)].x}', end=' ')
                    # 变量.x访问具体值
                    x[i,j,k] = result_x[(i, j, k)].x
        for i in range(0, args.app_num):
            for k in range(0, args.edge_num):
                for k_ in range(0, args.edge_num):
                    # name = 'result_y({},{},{})'.format(i, k, k_)
                    # print(f'{name}={result_y[(i, k, k_)].x}', end=' ')
                    y[i,k,k_] = result_y[(i, k, k_)].x
        for i in range(0, args.app_num):
            for j in range(0, args.inference_num):
                for k in range(0, args.edge_num):
                    # name = 'result_b({},{},{})'.format(i, j, k)
                    # print(f'{name}={result_b[(i, j, k)].x}', end=' ')
                    b[i,j,k] = result_b[(i, j, k)].x
        return x, y, b, model.objVal
    else:
        my_print("Optimization was stopped with status", model.status)
        import sys
        sys.exit(1)

def main():
    # 问题规模和初始化参数
    args = Params()

    # 设置优化问题为随机参数
    args.set_random()

    # 展示部分参数
    args.show_para()

    # 根据args生成优化问题，并求解优化问题，将求解后的结果返回
    x, y, b, _ = prob(args)

    # 展示结果
    for i in range(args.app_num):
        show_xyb(args, x, y, b, app=i)

if __name__ == '__main__':
    main()