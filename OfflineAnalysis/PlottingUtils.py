import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

def get_random_idx(array, size=40000, Repeat=False):
    rand_idx = np.random.choice(array[100:-100].index, size, replace=Repeat)
    return rand_idx

def plot_LDA(m,rand_idx,states=False,labels='states',alpha=0.2,size=5,linewidths=0,savefigure=False):
    LD_df = m.LD_df.copy()
    if states:
        if labels =='states':
            c_rules = m.state_df[labels].loc[rand_idx].apply(lambda x: m.colors[x])
        else:
            cm = plt.get_cmap('tab10')
            unique_labels = np.unique(m.state_df[labels])
            num = len(unique_labels)
            col_cycle = cycler(cycler('color', [cm(1.*i/num) for i in range(num)]))
            colors = dict(zip(unique_labels,col_cycle.by_key()["color"]))
            c_rules = m.state_df[labels].loc[rand_idx].apply(lambda x: colors[x])
    else:
        c_rules = 'k'
    if len(LD_df) != len(rand_idx):
        print ('selecting a subsample of input data to plot')
        LD_df = LD_df.loc[rand_idx]
    if LD_df.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(LD_df['LD1'],LD_df['LD2'], LD_df['LD3'],
                       c=c_rules,alpha=alpha, s=size,linewidths=linewidths)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
    else:
        plt.figure()
        ax.scatter(LD_df['LD1'],LD_df['LD2'],
                       c=c_rules,alpha=alpha, s=size,linewidths=linewidths)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
    if savefigure:
        plt.savefig(m.figureFolder+'LDA 3D states DPC labels {}_{}'.format(Folder[:6],File[:6]) + m.figure_tail, dpi=dpi)

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

