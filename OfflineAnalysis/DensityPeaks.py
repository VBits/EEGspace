from pydpc import Cluster
import numpy as np
import matplotlib.pyplot as plt

def get_dpc(m,rand_idx,savefigure=False):
    # build the density peak clusterer
    sample_data = np.ascontiguousarray(m.LD_df.loc[rand_idx].values)
    clu = Cluster(sample_data)
    plt.title('DPC', fontsize=15)
    if savefigure:
        plt.savefig(m.figureFolder+'Density peaks' + m.figure_tail)
    return clu

def dpc_cutoffs(clu,density,delta,savefigure=False):
    # decide the cutoffs for the clusters
    clu.assign(density,delta)
    plt.title('DPC boundaries', fontsize=15)
    if savefigure:
        plt.savefig(figureFolder+'Density peak boundaries' + m.figure_tail)
    return clu