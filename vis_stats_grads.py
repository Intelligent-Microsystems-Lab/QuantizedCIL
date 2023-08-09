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

track_layer_list = ['_convnet_conv_1_3x3',  # '_convnet_stage_1_2_conv_b',
                    '_convnet_stage_2_4_conv_a', '_convnet_stage_3_3_conv_a', '_fc']


def prep_plot(data, posi, posj, title, xlabel, ylabel, range_y):

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
  # ax[posi][posj].set_yscale('log')
  ax[posi][posj].plot(data[4:], lw=gen_linewidth)
  ax[posi][posj].set_ylim(range_y[0], range_y[1])
  # ax[posi][posj].hist(data, range = range_g, bins=50)


for stat_name in ['max', 'min', 'norm', 'mean']:

  fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(7.48031 * 2, 5. * 2),)

  i = 0
  for layer_name in track_layer_list:

    data = np.load('track_stats/23_07_31_13_00_cifar100_lwf_0'
                   + layer_name + '_' + stat_name + '.npy')
    range_y = [data.min(), data.max()]
    prep_plot(data, i, 0, 'Step 1', '', layer_name, range_y)

    data = np.load('track_stats/23_07_31_13_26_cifar100_lwf_1'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 1, 'Step 2', '', '', range_y)

    data = np.load('track_stats/23_07_31_13_51_cifar100_lwf_2'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 2, 'Step 3', '', '', range_y)

    data = np.load('track_stats/23_07_31_14_16_cifar100_lwf_3'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 3, 'Step 4', '', '', range_y)

    data = np.load('track_stats/23_07_31_14_41_cifar100_lwf_4'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 4, 'Step 5', '', '', range_y)

    i += 1

  plt.tight_layout()
  plt.savefig('figures/' + stat_name + '_grads_lwf.png',
              dpi=300, bbox_inches='tight')
  plt.close()


for stat_name in ['max', 'min', 'norm', 'mean']:

  fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(7.48031 * 2, 5. * 2),)

  i = 0
  for layer_name in track_layer_list:

    data = np.load('track_stats/23_07_31_12_56_cifar100_icarl_0'
                   + layer_name + '_' + stat_name + '.npy')
    range_y = [data.min(), data.max()]
    prep_plot(data, i, 0, 'Step 1', '', layer_name, range_y)

    data = np.load('track_stats/23_07_31_13_15_cifar100_icarl_1'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 1, 'Step 2', '', '', range_y)

    data = np.load('track_stats/23_07_31_13_36_cifar100_icarl_2'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 2, 'Step 3', '', '', range_y)

    data = np.load('track_stats/23_07_31_13_59_cifar100_icarl_3'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 3, 'Step 4', '', '', range_y)

    data = np.load('track_stats/23_07_31_14_20_cifar100_icarl_4'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 4, 'Step 5', '', '', range_y)

    i += 1

  plt.tight_layout()
  plt.savefig('figures/' + stat_name + '_grads_icarl.png',
              dpi=300, bbox_inches='tight')
  plt.close()


for stat_name in ['max', 'min', 'norm', 'mean']:

  fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(7.48031 * 2, 5. * 2),)

  i = 0
  for layer_name in track_layer_list:

    # if 'track_stats/23_07_31_12_50_cifar100_bic_0'+layer_name+'_'+stat_name+'.npy' == 'track_stats/23_07_31_12_49_cifar100_bic_0_convnet_stage_2_4_conv_a_max.npy':
    #   data = np.load('track_stats/23_07_31_12_50_cifar100_bic_0_convnet_stage_2_4_conv_a_max.npy')
    # else:
    try:
      data = np.load('track_stats/23_07_31_12_50_cifar100_bic_0'
                     + layer_name + '_' + stat_name + '.npy')
    except:
      data = np.load('track_stats/23_07_31_12_49_cifar100_bic_0'
                     + layer_name + '_' + stat_name + '.npy')
    range_y = [data.min(), data.max()]
    prep_plot(data, i, 0, 'Step 1', '', layer_name, range_y)

    try:
      data = np.load('track_stats/23_07_31_13_26_cifar100_bic_1'
                     + layer_name + '_' + stat_name + '.npy')
    except:
      data = np.load('track_stats/23_07_31_13_25_cifar100_bic_1'
                     + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 1, 'Step 2', '', '', range_y)

    data = np.load('track_stats/23_07_31_14_04_cifar100_bic_2'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 2, 'Step 3', '', '', range_y)

    data = np.load('track_stats/23_07_31_14_43_cifar100_bic_3'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 3, 'Step 4', '', '', range_y)

    data = np.load('track_stats/23_07_31_15_24_cifar100_bic_4'
                   + layer_name + '_' + stat_name + '.npy')
    prep_plot(data, i, 4, 'Step 5', '', '', range_y)

    i += 1

  plt.tight_layout()
  plt.savefig('figures/' + stat_name + '_grads_bic.png',
              dpi=300, bbox_inches='tight')
  plt.close()

# track_stats/23_07_31_12_50_cifar100_bic_0_convnet_conv_1_3x3_max.npy
# track_stats/23_07_31_12_49_cifar100_bic_0_convnet_conv_1_3x3_max.npy
# track_stats/23_07_31_12_50_cifar100_bic_0_convnet_stage_2_4_conv_a_max.npy



