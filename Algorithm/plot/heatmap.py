# 来自matplotlib官网
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

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

    title_font_dict = {
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

def main():
    app_num, inference_num, edge_num= 3, 4, 4
    app_label, infer_label, edge_label = [], [], []
    for i in range(app_num):
        app_label.append(f'App-{i}')
    for j in range(inference_num):
        infer_label.append(f'Infer-{j}')
    for i in range(edge_num):
        edge_label.append(f'Edge-{i}')
    title = "App and inferene loss distribution"

    loss_avg = 0.2
    loss_delta = 0.1
    loss_lb = loss_avg - loss_delta
    loss_ub = loss_avg + loss_delta

    req_avg = 8
    req_delta = 1
    req_lb = req_avg - req_delta
    req_ub = req_avg + req_delta

    model_mem_avg = 1
    model_mem_delta = 0.1
    model_mem_lb = model_mem_avg - model_mem_delta
    model_mem_ub = model_mem_avg + model_mem_delta

    xi_avg = 1
    xi_delta = 0.1
    xi_ub = xi_avg + xi_delta
    xi_lb = xi_avg - xi_delta

    loss = np.clip(
        np.random.normal(loss_avg, loss_delta, size=(app_num, inference_num)), 
        loss_lb, loss_ub
    ).round(decimals=2)

    loss = np.sort(loss, axis=1)

    req = np.round(
        np.clip(np.random.normal(req_avg, req_delta, size=(app_num, edge_num)), req_lb, req_ub)
    )

    model_mem = np.clip(
        np.random.normal(model_mem_avg, model_mem_delta, size=(app_num, inference_num)), 
        model_mem_lb, model_mem_ub
    ).round(decimals=2)

    model_mem = np.sort(model_mem, axis=1)[:, ::-1]

    xi = np.clip(
        np.random.normal(xi_avg, xi_delta, size=(app_num, inference_num)),
        xi_lb, xi_ub
    ).round(decimals=2)

    xi = np.sort(xi, axis=1)[:, ::-1]
    
    # fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    im, _ = heatmap(loss, app_label, infer_label, ax=ax, cmap='coolwarm', cbarlabel='Loss')
    annotate_heatmap(im)

    im, _ = heatmap(req, app_label, edge_label, ax=ax2, cmap='coolwarm', cbarlabel='Request Number')
    annotate_heatmap(im)

    im, _ = heatmap(xi, app_label, infer_label, ax=ax3, cmap='coolwarm', cbarlabel='Model Transmission')
    annotate_heatmap(im)

    im, _ = heatmap(model_mem, app_label, infer_label, ax=ax4, cmap='coolwarm', cbarlabel='Model Memory')
    annotate_heatmap(im)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()