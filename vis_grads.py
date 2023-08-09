import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter

import string
import pickle

import numpy as np
import pandas as pd
import copy
import itertools


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm as CM
from matplotlib.lines import Line2D


font_size = 14
font_size_aux = 10
gen_linewidth = 1.5

# Helvetica
plt.rc('font', family='Roboto', weight='bold')
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(7.48031 * 2, 5. * 2),)


track_layer_list = ['_convnet_conv_1_3x3', '_convnet_stage_1_2_conv_b',
                    '_convnet_stage_2_4_conv_a', '_convnet_stage_3_3_conv_a', '_fc']


def prep_plot(data, posi, posj, title, xlabel, ylabel, range_g):

  ax[posi][posj].set_title(
      title, {'fontsize': font_size, 'fontweight': 'bold', })

  for tick in ax[posi][posj].xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax[posi][posj].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax[posi][posj].spines['top'].set_visible(False)
  ax[posi][posj].spines['right'].set_visible(False)

  for axis in ['left', 'bottom']:
    ax[posi][posj].spines[axis].set_linewidth(gen_linewidth)

  ax[posi][posj].tick_params(width=gen_linewidth,)
  ax[posi][posj].tick_params(
      axis='both', which='major', labelsize=font_size_aux,)

  [label.set_fontweight('bold') for label in ax[posi][posj].get_yticklabels()]
  [label.set_fontweight('bold') for label in ax[posi][posj].get_xticklabels()]

  ax[posi][posj].set_ylabel(
      ylabel, {'fontsize': font_size_aux, 'fontweight': 'bold', })
  ax[posi][posj].set_xlabel(
      xlabel, {'fontsize': font_size_aux, 'fontweight': 'bold', })

  # data = data[data != 0]
  ax[posi][posj].set_yscale('log')
  ax[posi][posj].hist(data, range=range_g, bins=50)


data = np.load(
    'track_stats/23_07_25_16_44_cifar100_bic_0_convnet_conv_1_3x3.npy')
range_base = (data.min(), data.max())
prep_plot(data, 0, 0, 'Step 1', '', 'conv_1_3x3', range_base)

data = np.load(
    'track_stats/23_07_25_17_30_cifar100_bic_1_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 1, 'Step 2', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_18_18_cifar100_bic_2_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 2, 'Step 3', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_19_04_cifar100_bic_3_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 3, 'Step 4', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_19_51_cifar100_bic_4_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 4, 'Step 5', '', '', range_base)


data = np.load(
    'track_stats/23_07_25_16_45_cifar100_bic_0_convnet_stage_2_4_conv_a.npy')
range_base = (data.min(), data.max())
prep_plot(data, 1, 0, '', '', 'stage_2_4_conv_a', range_base)

data = np.load(
    'track_stats/23_07_25_17_30_cifar100_bic_1_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 1, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_18_18_cifar100_bic_2_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 2, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_19_05_cifar100_bic_3_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 3, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_19_52_cifar100_bic_4_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 4, '', '', '', range_base)


data = np.load(
    'track_stats/23_07_25_16_45_cifar100_bic_0_convnet_stage_3_3_conv_a.npy')
range_base = (data.min(), data.max())
prep_plot(data, 2, 0, '', '', 'stage_3_3_conv_a', range_base)

data = np.load(
    'track_stats/23_07_25_17_30_cifar100_bic_1_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 1, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_18_18_cifar100_bic_2_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 2, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_19_05_cifar100_bic_3_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 3, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_19_52_cifar100_bic_4_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 4, '', '', '', range_base)


data = np.load('track_stats/23_07_27_19_13_cifar100_bic_0_fc.npy')
range_base = (data.min(), data.max())
prep_plot(data, 3, 0, '', '', 'fc', range_base)

data = np.load('track_stats/23_07_27_19_33_cifar100_bic_1_fc.npy')
prep_plot(data, 3, 1, '', '', '', range_base)

data = np.load('track_stats/23_07_27_19_54_cifar100_bic_2_fc.npy')
prep_plot(data, 3, 2, '', '', '', range_base)

data = np.load('track_stats/23_07_27_20_16_cifar100_bic_3_fc.npy')
prep_plot(data, 3, 3, '', '', '', range_base)

data = np.load('track_stats/23_07_27_20_38_cifar100_bic_4_fc.npy')
prep_plot(data, 3, 4, '', '', '', range_base)


plt.tight_layout()
plt.savefig('figures/grads_bic.png', dpi=300, bbox_inches='tight')
plt.close()


# Helvetica
plt.rc('font', family='Roboto', weight='bold')
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(7.48031 * 2, 5. * 2),)


data = np.load(
    'track_stats/23_07_25_16_42_cifar100_lwf_0_convnet_conv_1_3x3.npy')
range_base = (data.min(), data.max())
prep_plot(data, 0, 0, 'Step 1', '', 'conv_1_3x3', range_base)

data = np.load(
    'track_stats/23_07_25_17_06_cifar100_lwf_1_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 1, 'Step 2', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_30_cifar100_lwf_2_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 2, 'Step 3', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_55_cifar100_lwf_3_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 3, 'Step 4', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_18_19_cifar100_lwf_4_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 4, 'Step 5', '', '', range_base)


data = np.load(
    'track_stats/23_07_25_16_42_cifar100_lwf_0_convnet_stage_2_4_conv_a.npy')
range_base = (data.min(), data.max())
prep_plot(data, 1, 0, '', '', 'stage_2_4_conv_a', range_base)

data = np.load(
    'track_stats/23_07_25_17_06_cifar100_lwf_1_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 1, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_30_cifar100_lwf_2_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 2, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_55_cifar100_lwf_3_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 3, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_18_19_cifar100_lwf_4_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 4, '', '', '', range_base)


data = np.load(
    'track_stats/23_07_25_16_42_cifar100_lwf_0_convnet_stage_3_3_conv_a.npy')
range_base = (data.min(), data.max())
prep_plot(data, 2, 0, '', '', 'stage_3_3_conv_a', range_base)

data = np.load(
    'track_stats/23_07_25_17_06_cifar100_lwf_1_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 1, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_31_cifar100_lwf_2_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 2, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_55_cifar100_lwf_3_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 3, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_18_19_cifar100_lwf_4_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 4, '', '', '', range_base)


data = np.load('track_stats/23_07_25_16_42_cifar100_lwf_0_fc.npy')
range_base = (data.min(), data.max())
prep_plot(data, 3, 0, '', '', 'fc', range_base)

data = np.load('track_stats/23_07_25_17_06_cifar100_lwf_1_fc.npy')
prep_plot(data, 3, 1, '', '', '', range_base)

data = np.load('track_stats/23_07_25_17_31_cifar100_lwf_2_fc.npy')
prep_plot(data, 3, 2, '', '', '', range_base)

data = np.load('track_stats/23_07_25_17_55_cifar100_lwf_3_fc.npy')
prep_plot(data, 3, 3, '', '', '', range_base)

data = np.load('track_stats/23_07_25_18_19_cifar100_lwf_4_fc.npy')
prep_plot(data, 3, 4, '', '', '', range_base)


plt.tight_layout()
plt.savefig('figures/grads_lwf.png', dpi=300, bbox_inches='tight')
plt.close()


# Helvetica
plt.rc('font', family='Roboto', weight='bold')
fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(7.48031 * 2, 5. * 2),)


data = np.load(
    'track_stats/23_07_25_16_39_cifar100_icarl_0_convnet_conv_1_3x3.npy')
range_base = (data.min(), data.max())
prep_plot(data, 0, 0, 'Step 1', '', 'conv_1_3x3', range_base)

data = np.load(
    'track_stats/23_07_25_16_58_cifar100_icarl_1_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 1, 'Step 2', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_17_cifar100_icarl_2_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 2, 'Step 3', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_37_cifar100_icarl_3_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 3, 'Step 4', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_58_cifar100_icarl_4_convnet_conv_1_3x3.npy')
prep_plot(data, 0, 4, 'Step 5', '', '', range_base)


data = np.load(
    'track_stats/23_07_25_16_39_cifar100_icarl_0_convnet_stage_2_4_conv_a.npy')
range_base = (data.min(), data.max())
prep_plot(data, 1, 0, '', '', 'stage_2_4_conv_a', range_base)

data = np.load(
    'track_stats/23_07_25_16_58_cifar100_icarl_1_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 1, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_18_cifar100_icarl_2_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 2, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_38_cifar100_icarl_3_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 3, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_58_cifar100_icarl_4_convnet_stage_2_4_conv_a.npy')
prep_plot(data, 1, 4, '', '', '', range_base)


data = np.load(
    'track_stats/23_07_25_16_40_cifar100_icarl_0_convnet_stage_3_3_conv_a.npy')
range_base = (data.min(), data.max())
prep_plot(data, 2, 0, '', '', 'stage_3_3_conv_a', range_base)

data = np.load(
    'track_stats/23_07_25_16_58_cifar100_icarl_1_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 1, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_18_cifar100_icarl_2_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 2, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_38_cifar100_icarl_3_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 3, '', '', '', range_base)

data = np.load(
    'track_stats/23_07_25_17_58_cifar100_icarl_4_convnet_stage_3_3_conv_a.npy')
prep_plot(data, 2, 4, '', '', '', range_base)


data = np.load('track_stats/23_07_25_16_40_cifar100_icarl_0_fc.npy')
range_base = (data.min(), data.max())
prep_plot(data, 3, 0, '', '', 'fc', range_base)

data = np.load('track_stats/23_07_25_16_58_cifar100_icarl_1_fc.npy')
prep_plot(data, 3, 1, '', '', '', range_base)

data = np.load('track_stats/23_07_25_17_18_cifar100_icarl_2_fc.npy')
prep_plot(data, 3, 2, '', '', '', range_base)

data = np.load('track_stats/23_07_25_17_38_cifar100_icarl_3_fc.npy')
prep_plot(data, 3, 3, '', '', '', range_base)

data = np.load('track_stats/23_07_25_17_58_cifar100_icarl_4_fc.npy')
prep_plot(data, 3, 4, '', '', '', range_base)


plt.tight_layout()
plt.savefig('figures/grads_icarl.png', dpi=300, bbox_inches='tight')
plt.close()