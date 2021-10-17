import numpy as np
import matplotlib.pyplot as plt

def get_random_idx(array, size=40000, Repeat=False):
    rand_idx = np.random.choice(len(array), size, replace=Repeat)
    return rand_idx

def plot_LDA(LD_df,rand_idx,states=None,colors=mh.colors,alpha=0.2,size=5,linewidths=0):
    if states is None:
        c_rules = 'k'
    else:
        c_rules = LD_df['states'][rand_idx].apply(lambda x: colors[x])
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

plot_LDA(LD_df_test,rand_idx)
