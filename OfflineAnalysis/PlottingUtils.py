import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_random_idx(array, size=40000, Repeat=False):
    rand_idx = np.random.choice(len(array), size, replace=Repeat)
    return rand_idx

def plot_LDA(m,rand_idx,states=False,alpha=0.2,size=5,linewidths=0,savefigure=False):
    if states:
        c_rules = m.state_df['states'][rand_idx].apply(lambda x: m.colors[x])
    else:
        c_rules = 'k'
    if len(m.LD_df) != len(rand_idx):
        print ('selecting a subsample of input data to plot')
        m.LD_df = m.LD_df.iloc[rand_idx]
    if m.LD_df.shape[1] == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(m.LD_df['LD1'],m.LD_df['LD2'], m.LD_df['LD3'],
                       c=c_rules,alpha=alpha, s=size,linewidths=linewidths)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
    else:
        plt.figure()
        ax.scatter(m.LD_df['LD1'],m.LD_df['LD2'],
                       c=c_rules,alpha=alpha, s=size,linewidths=linewidths)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
    if savefigure:
        plt.savefig(figureFolder+'LDA 3D states DPC labels {}_{}'.format(Folder[:6],File[:6]) + m.figure_tail, dpi=dpi)

def plot_EEG(m, File hide_figure=True):
    if hide_figure:
        print('displaying')
        matplotlib.use('Agg')
    plt.figure()
    plt.plot(m.EEG_data)
    plt.title('{}'.format(m.Ch_name))
    plt.ylabel(m.Ch_units)
    plt.ylim(1000,-1000)
    plt.savefig(figureFolder+'{}_{}'.format(m.Ch_name,File[:6]) + m.figure_tail)
    # matplotlib.use('Qt5Agg')

plot_LDA(LD_df_test,rand_idx)
