import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler
import pandas as pd
from OfflineAnalysis.GeneralUtils import query_yes_no
from OfflineAnalysis import Config as OfflineConfig
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def get_cmap_colors(labels,rand_idx,cmap='tab10'):
    cm = plt.get_cmap(cmap)
    unique_labels = np.unique(labels)
    num = len(unique_labels)
    col_cycle = cycler(cycler('color', [cm(1. * i / num) for i in range(num)]))
    colors = dict(zip(unique_labels, col_cycle.by_key()["color"]))
    c_data = labels.loc[rand_idx].apply(lambda x: colors[x])
    return c_data


def plot_LDA(m, rand_idx, labels=None, alpha=0.2, size=5, linewidths=0, savefigure=None):
    LD_df = m.LD_df.copy()

    if labels is None:
        print(1)
        c_data = 'k'
        figure_title = OfflineConfig.lda_figure_title_no_labels
    elif isinstance(labels, (pd.core.series.Series, pd.core.frame.DataFrame)):
        print('labels are provided in a DataFrame')
        if "SWS" in labels.values:
            c_data = labels.loc[rand_idx].apply(lambda x: m.colors[x])
            figure_title = OfflineConfig.lda_figure_title_state_labels
        else:
            c_data = get_cmap_colors(labels, rand_idx, 'tab10')
            figure_title = OfflineConfig.lda_figure_title_dpc_labels
    elif isinstance(labels, np.ndarray):
        print('labels are provided in a numpy array. Use a pandas dataframe with timestamps instead.')
        return None

    if len(LD_df) != len(rand_idx):
        print('selecting a subsample of input data to plot')
        LD_df = LD_df.loc[rand_idx]

    if LD_df.shape[1] == 3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(LD_df['LD1'], LD_df['LD2'], LD_df['LD3'],
                   c=c_data, alpha=alpha, s=size, linewidths=linewidths)
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
        ax.set_zlabel('LD3')
    else:
        plt.figure()
        plt.scatter(LD_df['LD1'], LD_df['LD2'],
                    c=c_data, alpha=alpha, s=size, linewidths=linewidths)
        plt.xlabel('LD1')
        plt.ylabel('LD2')

    plt.show(block=False)

    if savefigure is None:
        savefigure = query_yes_no("Do you want to save plot? Please respond with yes or no")

    if savefigure:
        plt.savefig(m.figureFolder + figure_title + m.figure_tail, dpi=OfflineConfig.dpi)


def plot_DPA_LDA(m, rand_idx, est, alpha=0.6, size=4, linewidths=0, savefigure=None, remapped=False, view_angles=(90, -67)):
    labels = est.labels_
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Choose colormap based on remapped flag and cluster count
    if remapped:
        distinct_colors = plt.get_cmap('Set2').colors
    elif n_clusters <= 8:
        distinct_colors = plt.get_cmap('tab10').colors[:n_clusters]
    elif n_clusters <= 20:
        # Reorder tab20b colors to improve distinctiveness
        tab20b_colors = list(plt.get_cmap('tab20b').colors)
        group_size = 4
        num_groups = 5
        reordered_colors = []
        for i in range(group_size):
            for j in range(num_groups):
                reordered_colors.append(tab20b_colors[j * group_size + i])
        distinct_colors = reordered_colors[:n_clusters]
    else:
        # Use a continuous colormap for many clusters
        cmap = plt.get_cmap('gist_ncar')
        distinct_colors = [cmap(i / n_clusters) for i in range(n_clusters)]

    cmap = ListedColormap(distinct_colors)
    color_map = {label: cmap(i % len(distinct_colors)) for i, label in enumerate(unique_labels)}
    colors = [color_map[label] for label in labels]

    ax = plt.figure().add_subplot(projection='3d')

    # Apply fixed 3D view if specified
    if view_angles is not None:
        elev, azim = view_angles
        ax.view_init(elev=elev, azim=azim)

    scatter = ax.scatter(
        m.LD_df.loc[rand_idx].values[:, 0],
        m.LD_df.loc[rand_idx].values[:, 1],
        m.LD_df.loc[rand_idx].values[:, 2],
        alpha=alpha, s=size, linewidths=linewidths, c=colors
    )

    ax.set_xlabel('LD1')
    ax.set_ylabel('LD2')
    ax.set_zlabel('LD3')

    patches = [mpatches.Patch(color=color_map[label], label=f'Cluster {label}') for label in unique_labels]
    ax.legend(handles=patches, loc='upper right', title='DPA cluster')

    plt.show(block=False)

    if savefigure is None:
        savefigure = query_yes_no("Do you want to save plot? Please respond with yes or no")

    if savefigure:
        plt.savefig(m.figureFolder + OfflineConfig.lda_figure_title_dpc_labels + m.figure_tail, dpi=OfflineConfig.dpi)


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

