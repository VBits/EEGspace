import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
import pandas as pd

def get_cmap_colors(labels,rand_idx,cmap='tab10'):
    cm = plt.get_cmap(cmap)
    unique_labels = np.unique(labels)
    num = len(unique_labels)
    col_cycle = cycler(cycler('color', [cm(1. * i / num) for i in range(num)]))
    colors = dict(zip(unique_labels, col_cycle.by_key()["color"]))
    c_data = labels.loc[rand_idx].apply(lambda x: colors[x])
    return c_data

def plot_LDA(m,rand_idx,labels=None,alpha=0.2,size=5,linewidths=0,savefigure=False):
    LD_df = m.LD_df.copy()
    if labels is None:
        print(1)
        c_data = 'k'
    elif isinstance(labels, (pd.core.series.Series,pd.core.frame.DataFrame)):
        print('labels are provided in a DataFrame')
        if "SWS" in labels.values:
            c_data = labels.loc[rand_idx].apply(lambda x: m.colors[x])
        else:
            c_data = get_cmap_colors(labels, rand_idx, 'tab10')
    elif isinstance(labels, (np.ndarray)):
        print('labels are provided in a numpy array. Use a pandas dataframe with timestamps instead.')
        return None
        # if "SWS" in labels:
        #     c_data = labels[rand_idx].apply(lambda x: m.colors[x])
        # else:
        #     c_data = get_cmap_colors(labels, rand_idx, 'tab10')
    if len(LD_df) != len(rand_idx):
        print ('selecting a subsample of input data to plot')
        LD_df = LD_df.loc[rand_idx]
    if LD_df.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(LD_df['LD1'],LD_df['LD2'], LD_df['LD3'],
                       c=c_data,alpha=alpha, s=size,linewidths=linewidths)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
    else:
        plt.figure()
        plt.scatter(LD_df['LD1'],LD_df['LD2'],
                       c=c_data,alpha=alpha, s=size,linewidths=linewidths)
        plt.xlabel('LD1')
        plt.ylabel('LD2')
    if savefigure:
        plt.savefig(m.figureFolder+'LDA 3D states DPC labels {}_{}'.format(Folder[:6],File[:6]) + m.figure_tail, dpi=dpi)

def plot_DPA_LDA(m, rand_idx, est, alpha=0.6, size=4, linewidths=0, savefigure=False):
    fig = plt.figure()
    ax = Axes3D(fig)
    scatter = ax.scatter(m.LD_df.loc[rand_idx].values[:, 0], m.LD_df.loc[rand_idx].values[:, 1],
               m.LD_df.loc[rand_idx].values[:, 2],
               alpha=alpha, s=size,linewidths=linewidths, c=est.labels_, cmap='Accent')
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    legend1 = ax.legend(*scatter.legend_elements(num=len(np.unique(est.labels_))-1),
                        loc="upper right", title="DPA cluster")
    ax.add_artist(legend1)
    if savefigure:
        plt.savefig(m.figureFolder + 'LDA 3D states DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail,dpi=dpi)


def plot_EEG(m, File, hide_figure=True):
    if hide_figure:
        print('saving without displaying')
        matplotlib.use('Agg')
    plt.figure()
    plt.plot(m.EEG_data)
    plt.title('{}'.format(m.Ch_name))
    plt.ylabel(m.Ch_units)
    plt.ylim(1000,-1000)
    plt.savefig(m.figureFolder+'{}_{}'.format(m.Ch_name,File[:6]) + m.figure_tail)
    matplotlib.use('Qt5Agg')

