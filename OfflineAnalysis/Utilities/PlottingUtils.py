import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
import pandas as pd
from OfflineAnalysis.Utilities.GeneralUtils import query_yes_no
import OfflineAnalysis.Config as OfflineConfig
import sys
def get_cmap_colors(labels,rand_idx,cmap='tab10'):
    cm = plt.get_cmap(cmap)
    unique_labels = np.unique(labels)
    num = len(unique_labels)
    col_cycle = cycler(cycler('color', [cm(1. * i / num) for i in range(num)]))
    colors = dict(zip(unique_labels, col_cycle.by_key()["color"]))
    c_data = labels.loc[rand_idx].apply(lambda x: colors[x])
    return c_data

def plot_LDA(m,rand_idx,labels=None,alpha=0.2,size=5,linewidths=0):
    # plt.ion()
    # plt.show(block=False)
    LD_df = m.LD_df.copy()
    if labels is None:
        print(1)
        c_data = 'k'
        # figure_title = OfflineConfig.lda_figure_title_no_labels
    elif isinstance(labels, (pd.core.series.Series,pd.core.frame.DataFrame)):
        print('labels are provided in a DataFrame')
        if "SWS" in labels.values:
            c_data = labels.loc[rand_idx].apply(lambda x: m.colors[x])
            # figure_title = OfflineConfig.lda_figure_title_state_labels
        else:
            c_data = get_cmap_colors(labels, rand_idx, 'tab10')
            # figure_title = OfflineConfig.lda_figure_title_dpc_labels
    elif isinstance(labels, (np.ndarray)):
        print('labels are provided in a numpy array. Use a pandas dataframe with timestamps instead.')
        return None
    if len(LD_df) != len(rand_idx):
        print ('selecting a subsample of input data to plot')
        LD_df = LD_df.loc[rand_idx]
    if LD_df.shape[1] == 3:
        ax = plt.figure().add_subplot(projection='3d')
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
    # plt.draw()
    # plt.pause(10)
    plt.show()
    # savefigure = query_yes_no("Do you want to save plot? Please respond with yes or no")
    # if savefigure:
    #     plt.savefig(m.figureFolder + figure_title + m.figure_tail,dpi=OfflineConfig.dpi)

def create_new_plot_LDA_multiprocessing_version(LD_df,rand_idx, colors, labels=None,DPA=False,alpha=0.2,size=5,linewidths=0):
    if DPA:
        c_data = labels
    else:
        if labels is None:
            print(1)
            c_data = 'k'
            # figure_title = OfflineConfig.lda_figure_title_no_labels
        elif isinstance(labels, (pd.core.series.Series,pd.core.frame.DataFrame)):
            print('labels are provided in a DataFrame')
            if "SWS" in labels.values:
                c_data = labels.loc[rand_idx].apply(lambda x: colors[x])
                # figure_title = OfflineConfig.lda_figure_title_state_labels
            else:
                c_data = get_cmap_colors(labels, rand_idx, 'tab10')
                # figure_title = OfflineConfig.lda_figure_title_dpc_labels
        elif isinstance(labels, (np.ndarray)):
            print('labels are provided in a numpy array. Use a pandas dataframe with timestamps instead.')
            return None
    if len(LD_df) != len(rand_idx):
        print ('selecting a subsample of input data to plot')
        LD_rand = LD_df.loc[rand_idx]
    else:
        LD_rand = LD_df
    if DPA:
        if LD_rand.shape[1] == 3:
            ax = plt.figure().add_subplot(projection='3d')
            scatter = ax.scatter(LD_rand.values[:, 0], LD_rand.loc[rand_idx].values[:, 1],
                                 LD_rand.values[:, 2],
                                 alpha=alpha, s=size, linewidths=linewidths, c=colors, cmap='tab20b')
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            ax.set_zlabel('LD3')
            legend1 = ax.legend(*scatter.legend_elements(num=len(np.unique(labels)) - 1),
                                loc="upper right", title="DPA cluster")
            ax.add_artist(legend1)
            plt.show(block=False)
        else:
            ax = plt.figure().add_subplot()
            scatter = ax.scatter(LD_rand.values[:, 0], LD_rand.loc[rand_idx].values[:, 1],
                                 alpha=alpha, s=size, linewidths=linewidths, c=c_data, cmap='tab20b')
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            legend1 = ax.legend(*scatter.legend_elements(num=len(np.unique(labels)) - 1),
                                loc="upper right", title="DPA cluster")
            ax.add_artist(legend1)
            plt.show(block=False)
    else:
        if LD_rand.shape[1] == 3:
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(LD_rand['LD1'],LD_rand['LD2'], LD_rand['LD3'],
                           c=c_data,alpha=alpha, s=size,linewidths=linewidths)
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            ax.set_zlabel('LD3')
        else:
            plt.figure()
            plt.scatter(LD_rand['LD1'],LD_rand['LD2'],
                           c=c_data,alpha=alpha, s=size,linewidths=linewidths)
            plt.xlabel('LD1')
            plt.ylabel('LD2')
    plt.draw()
    plt.pause(1)
    plt.show()

def plot_LDA_multiprocessing_version(LD_df,rand_idx, figure_folder, figure_tail, colors, input_queue, labels=None, DPA=False, alpha=0.2,size=5,linewidths=0):
    sys.stdout.write("testing output 1")
    sys.stdout.flush()
    create_new_plot_LDA_multiprocessing_version(LD_df,rand_idx, colors, labels, DPA, alpha, size, linewidths)
    sys.stdout.write("testing output 3")
    while True:
        if not input_queue.empty():
            sys.stdout.write("got input")
            input = input_queue.get()
            if input[0] == "save_plot":
                sys.stdout.write(figure_folder + input[1] + figure_tail)
                plt.savefig(figure_folder + input[1] + figure_tail,dpi=OfflineConfig.dpi)
                plt.figure()
            elif input[0] == "no_save_plot":
                continue
            else:
                create_new_plot_LDA_multiprocessing_version(input[0], input[1], colors, input[2], input[3], alpha, size, linewidths)

def plot_DPA_LDA(m, rand_idx, est, alpha=0.6, size=4, linewidths=0):
    ax = plt.figure().add_subplot(projection='3d')
    scatter = ax.scatter(m.LD_df.loc[rand_idx].values[:, 0], m.LD_df.loc[rand_idx].values[:, 1],
               m.LD_df.loc[rand_idx].values[:, 2],
               alpha=alpha, s=size,linewidths=linewidths, c=est.labels_, cmap='tab20b')
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    legend1 = ax.legend(*scatter.legend_elements(num=len(np.unique(est.labels_))-1),
                        loc="upper right", title="DPA cluster")
    ax.add_artist(legend1)
    plt.show(block=False)
    # savefigure = query_yes_no("Do you want to save plot? Please respond with yes or no")
    # if savefigure:
    #     plt.savefig(m.figureFolder + OfflineConfig.lda_figure_title_dpc_labels + m.figure_tail,dpi=OfflineConfig.dpi)

def plot_EEG(m, File, hide_figure=True):
    if hide_figure:
        print('saving without displaying')
        matplotlib.use('Agg')
    plt.figure()
    plt.plot(m.EEG_data)
    plt.title('{}'.format(m.Ch_name))
    plt.ylabel(m.Ch_units)
    plt.ylim(1000,-1000)
    plt.savefig(m.figureFolder+ OfflineConfig.eeg_figure_title + m.figure_tail)
    matplotlib.use('Qt5Agg')

def plot_outliers(m,rand_idx,outlier_model):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(m.LD_df['LD1'].loc[rand_idx],m.LD_df['LD2'].loc[rand_idx],m.LD_df['LD3'].loc[rand_idx],c=outlier_model.labels_,cmap='Dark2_r',alpha=0.5, s=5)
    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')
    plt.title('LDA')
    # plt.savefig(m.figureFolder+'LDA and labeled outliers' + m.figure_tail, dpi=dpi)

def savefigure_function(m, figure_title):
    """You can call this function and ask if you want to save the active figure"""
    savefigure = query_yes_no("Do you want to save plot? Please respond with yes or no")
    if savefigure:
        plt.savefig(m.figureFolder + figure_title + m.figure_tail,dpi=OfflineConfig.dpi)

def savefigure_function_multiprocessing_version(queue, figure_title):
    """You can call this function and ask if you want to save the active figure"""
    savefigure = query_yes_no("Do you want to save plot? Please respond with yes or no")
    if savefigure:
        print("putting in queue")
        queue.put(("save_plot", figure_title))
        return
    queue.put(("no_save_plot",))

