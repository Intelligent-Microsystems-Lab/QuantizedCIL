import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy

figsize = (8.48, 3.5)
linewidth = 1.5
fontsize = 13

# with sns.axes_style('darkgrid'):
fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=False, squeeze=True,)
                         # gridspec_kw={'height_ratios': [1,1]})

gbins = 15

colors = ["purple", "teal", "grey"]

axes[0,0].set_title('Weights 4-bit', fontsize=fontsize+1)

w_og = np.load('w.npy')
# w_og /= np.abs(w_og).max() * 15
# import pdb; pdb.set_trace()
axes[1,0].hist(w_og.flatten(), bins = gbins, range=(-7,7), color = "purple", edgecolor='black', linewidth=.9, alpha=.5)
axes[0,0].set_yscale('log')
axes[0,0].set_ylabel('Count')
axes[1,0].set_ylabel('Count')

axes[1,0].set_xlabel('Range')
axes[1,1].set_xlabel('Range')
axes[1,2].set_xlabel('Range')

w_og = np.load('w_nohad.npy')
# w_og /= np.abs(w_og).max() * 15
axes[0,0].hist(w_og.flatten(), bins = gbins, range=(-7,7), color = "teal", edgecolor='black', linewidth=.9, alpha=.5)
axes[1,0].set_yscale('log')

# w_og = np.load('w_noq.npy')
# w_og /= np.abs(w_og).max()
# axes[2,0].hist(w_og.flatten(), bins = gbins, range=(-1,1), color = "grey", edgecolor='black', linewidth=.9, alpha=.5)
# axes[2,0].set_yscale('log')


axes[0,1].set_title('Errors 4-bit', fontsize=fontsize+1)

w_og = np.load('g.npy')
# w_og /= np.abs(w_og).max() * 15
axes[1,1].hist(w_og.flatten(), bins = gbins, range=(-7,7), color = "purple", edgecolor='black', linewidth=.9, alpha=.5)
axes[0,1].set_yscale('log')

w_og = np.load('g_nohad.npy')
# w_og /= np.abs(w_og).max() * 15
axes[0,1].hist(w_og.flatten(), bins = gbins, range=(-7,7), color = "teal", edgecolor='black', linewidth=.9, alpha=.5)
axes[1,1].set_yscale('log')

rarrow = axes[0,1].annotate("", xy=(3, 4e0), xytext=(7, 4e0),
            arrowprops=dict(arrowstyle="<->", color='red'), label='Underutilized Range')

axes[0,1].annotate("", xy=(-3, 4e0), xytext=(0, 4e0),
            arrowprops=dict(arrowstyle="<->", color='red'))

axes[0,2].annotate("", xy=(-127, 2e0), xytext=(-50, 2e0),
            arrowprops=dict(arrowstyle="<->", color='red'))

axes[0,2].annotate("", xy=(50, 2e0), xytext=(127, 2e0),
            arrowprops=dict(arrowstyle="<->", color='red'))

# w_og = np.load('g_noq.npy')
# w_og /= np.abs(w_og).max()
# axes[2,1].hist(w_og.flatten(), bins = gbins, range=(-1,1), color = "grey", edgecolor='black', linewidth=.9, alpha=.5)
# axes[2,1].set_yscale('log')




axes[0,2].set_title('Accm. 8-bit', fontsize=fontsize+1)

gbins = 128

# w_og = np.load('ginp_noq.npy')
# w_og /= np.abs(w_og).max()
# base = w_og
# import pdb; pdb.set_trace()
# axes[2,2].hist(w_og.flatten(), bins = gbins, range=(-1,1), color = "grey", edgecolor='black', linewidth=.9, alpha=.5)
# axes[2,2].set_yscale('log')

w_og = np.load('ginp.npy') # * 3.3379e-08
# w_og /= np.abs(w_og).max()
# kl = scipy.stats.entropy(np.histogram(w_og.flatten(), bins=gbins, range=(-127,127))[0],np.histogram(base.flatten(), bins=gbins, range=(-1,1))[0])
axes[1,2].hist(w_og.flatten(), bins = gbins, range=(-127,127), color = "purple", edgecolor='black', linewidth=.02, alpha=.5)
# axes[0,2].set_title(f'KL to FP {np.round(kl,3)}', fontsize=fontsize+1)
axes[0,2].set_yscale('log')

w_og = np.load('ginp_nohad.npy')
# w_og /= np.abs(w_og).max()
# kl = scipy.stats.entropy(np.histogram(w_og.flatten(), bins=gbins, range=(-127,127))[0],np.histogram(base.flatten(), bins=gbins, range=(-1,1))[0])
axes[0,2].hist(w_og.flatten(), bins = gbins, range=(-127,127), color = "teal", edgecolor='black', linewidth=.02, alpha=.5)
# axes[1,2].set_title(f'KL to FP {np.round(kl,3)}', fontsize=fontsize+1)
axes[1,2].set_yscale('log')

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p



legend_elements = [
                   Patch(facecolor='purple', edgecolor='k',
                         label='Hadamard Quantization (Ours)', alpha=.5),
                   Line2D([0], [0], color='r', lw=linewidth, label= 'Underutilized Range'),
                   Patch(facecolor='teal', edgecolor='k',
                         label='Standard Quantization',alpha=.5),
                   
                   # Patch(facecolor='grey', edgecolor='k',
                   #      label='FP')
]
fig.legend(handles = legend_elements, loc='lower center', bbox_to_anchor=(0.4, -0.12),
            ncol=2, fontsize=fontsize-2,  frameon=False,)


# # for i, axs in enumerate(axes):
# for ax in axes:
#     for tick in ax.xaxis.get_major_ticks():
#         tick.label1.set_fontweight('bold')
#     for tick in ax.yaxis.get_major_ticks():
#         tick.label1.set_fontweight('bold')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     [label.set_fontweight('bold') for label in ax.get_yticklabels()]
#     # if i == 0:
#     #     ax.set_ylim(ymin=0, ymax=100)
#     # else:
#     #     ax.set_ylim(ymin=0, ymax=50)



for i, axs in enumerate(axes):
    for ax in axs:
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        [label.set_fontweight('bold') for label in ax.get_yticklabels()]
        # if i == 0:
        #     ax.set_ylim(ymin=0, ymax=100)
        # else:
        #     ax.set_ylim(ymin=0, ymax=50)




# second part of figure


################
# HOLD ONE OUT
################
axes[0,3].set_title('Quant. Ablation', fontsize=fontsize+1)

# axes[0,3].plot( [79.69, 68.614, 58.452, 49.638, 42.92], label='fp')
# axes[0,3].plot([69.96, 56.544, 39.27, 10.03, 4.762], label='F: W')
# axes[0,3].plot([71.65, 59.314, 47.068, 31.95, 25.19], label='F: A')
axes[0,3].plot([69.62, 56.796, 44, 24.13, 15.498], marker='x', label='B1: W')
axes[0,3].plot([72.48, 61.58, 50.248, 41.348, 34.5], marker='x', label='B1: G')
axes[0,3].plot([74.67, 65.178, 54.802, 45.854, 38.266], marker='x', label='B2: A')
axes[0,3].plot([72.62, 61.636, 50.174, 40.84, 34.316], marker='x', label='B2: G')


axes[0,3].legend( loc='lower center', bbox_to_anchor=(0.23, -0.0),
            ncol=1, fontsize=fontsize-6,  frameon=True)


axes[0,3].set_ylabel('Accuracy')

#################
# Accm Quant
#################

# PAMAP - iCaRL
# axes[1,3].set_title('Accumulator Ablation', fontsize=fontsize+1)
# axes[1,3].plot( [98.81, 93.91,   87.3,    86.73,   85.97,   82.63], label = 'FP')
axes[1,3].plot( [98.21,   94.42,   82.09,   80.6,    82.78,   80.53], marker='x', label = 'F: Accm.')
axes[1,3].plot( [95.24,   40.86,   28.17,   24.82,   15.33,   0.95], marker='x', label = 'B1: Accm.')
axes[1,3].plot( [91.67,   40.1,    15.3,    28.67,   22.17,   19.68], marker='x', label = 'B2: Accm.')

axes[1,3].legend( loc='lower center', bbox_to_anchor=(0.65, 0.29),
            ncol=1, fontsize=fontsize-6,  frameon=True)
axes[1,3].set_yticks([0, 25,50,75])

axes[1,3].set_ylabel('Accuracy')
axes[1,3].set_xlabel('Tasks')

plt.tight_layout()

# save fig
fig.savefig('had_fig.pdf', bbox_inches='tight')
