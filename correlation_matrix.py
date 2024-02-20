import matplotlib as mpl
import matplotlib.font_manager as fm

import torch
import torch.nn.functional as F

import pickle
# import spiking_learning as sl
# import optax
import numpy as np
from sklearn.decomposition import PCA

# from jax import random
import matplotlib.pyplot as plt
import os
import seaborn as sb
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.data_manager import DataManager
from torch.utils.data import DataLoader

from torch.nn import functional as F

import torch.nn as nn

args_in_dim = 405
dset = ['dsads'][0]
tech = ['icarl', "bic"][1]
zoom = False
if zoom:
  nr_tasks = 2
else:
  if dset == 'dsads':
    nr_tasks = 10
  else:
    nr_tasks = 6


def get_nr_classes(task):
  return 2*(task+1) if task < 9 else 11

# nr_classes = get_nr_classes(task_to_display)
# dsads bic noq 80.35/9.82 ours 16 82.19/4.39 ours 8 70.88/18.51 luq 16 71.14/14.39 luq 8 33.77/21.49
# dsads icarl noq 80.35/9.82 ours 16  ours 8 luq 16 78.86/20.61 luq 8 52.72/44.39
model_files = [f'logs/{dset}/{tech}/weights/fcnet_noq_accbits_16_1994.npy',
               f'logs/{dset}/{tech}/weights/fcnet_ours_accbits_16_1994.npy',
               f'logs/{dset}/{tech}/weights/fcnet_luq_corrected_accbits_16_1994.npy',
               f'logs/{dset}/{tech}/weights/fcnet_ours_accbits_8_1994.npy',
               f'logs/{dset}/{tech}/weights/fcnet_luq_corrected_accbits_8_1994.npy',
                  ]
names = ['NoQ','HDQT (Ours) 16 Bits','LUQ 16 Bits','HDQT (Ours) 8 Bits',
        'LUQ 8 Bits',
        ]

def load_params(file='def', task=0, step=0):

    if file == 'def':
      data_s = np.load(model_files[default_file], allow_pickle=True)      
    else:
      data_s = np.load(file, allow_pickle=True)  
    data_s = np.reshape(data_s,(-1))[0]

    all_params = []
    backbone_l1 = None
    backbone_l2 = None
    fc = None
    for k,v in data_s[task][step].items():
      if 'hadamard' in k:
        continue
      if 'alpha' in k:
        continue
      if 'beta' in k:
        continue
      if 'grad' in k:
        continue
      if 'init' in k:
        continue
      # print(k)
      all_params.append(np.array(v.flatten().cpu().numpy()))
      if 'backbone.net.0' in k:
        backbone_l1 = np.array(v.flatten().cpu().numpy())
      if 'backbone.net.1' in k:
        backbone_l2 = np.array(v.flatten().cpu().numpy())
      if 'fc' in k:
        fc = np.array(v.flatten().cpu().numpy())
    

    return np.concatenate(all_params), backbone_l1, backbone_l2, fc, data_s[task][step].keys()

# calculate correlation matrices of the weights of 2 files models between tasks task at step 90
def calculate_correlation_matrices(file1, step=90):
  all_correlations = {}
  all_correlations_matrix = np.zeros((nr_tasks, nr_tasks))
  all_correlations_b1_matrix = np.zeros((nr_tasks, nr_tasks))
  all_correlations_b2_matrix = np.zeros((nr_tasks, nr_tasks))
  all_correlations_fc_matrix = np.zeros((nr_tasks, nr_tasks))
  all_keys = []
  for task in range(nr_tasks):
    for task2 in range(nr_tasks):
      
      params1, backbone_l1, backbone_l2, fc, keys = load_params(file1, task, step)
      params2, backbone_l12, backbone_l22, fc2,_ = load_params(file1, task2, step)

      all_keys = keys
      min_len = min(len(params1), len(params2))
      corr = np.corrcoef(params1[:min_len], params2[:min_len])
      corr_b1 = np.corrcoef(backbone_l1, backbone_l12)
      corr_b2 = np.corrcoef(backbone_l2, backbone_l22)
      min_len = min(len(fc), len(fc2))
      corr_fc = np.corrcoef(fc[:min_len], fc2[:min_len])

      all_correlations[(task, task2)] = corr[0][1]
      all_correlations_matrix[task, task2] = corr[0][1]
      all_correlations_b1_matrix[task, task2] = corr_b1[0][1]
      all_correlations_b2_matrix[task, task2] = corr_b2[0][1]
      all_correlations_fc_matrix[task, task2] = corr_fc[0][1]
  return all_correlations, all_correlations_matrix, all_correlations_b1_matrix, all_correlations_b2_matrix, all_correlations_fc_matrix, all_keys

# plot the heatmap of the correlation matrix
def plot_correlation_matrix(correlation_matrix, keys, name):
  plt.figure(figsize=(10, 10))
  sb.heatmap(correlation_matrix, annot=False, cmap='coolwarm', xticklabels=keys, yticklabels=keys)
  plt.title(name)
  plt.savefig(f"figures/correlation_matrices/{dset}/{tech}/{dset}_{tech}_corr_{name}.png")
  plt.close()
# def plot_two_correlation_matrices(corr_matrix1, corr_matrix2, keys, name1, name2):
#   import pdb; pdb.set_trace()
#   fig, ax = plt.subplots(1, 2, figsize=(20, 10))
#   sb.heatmap(corr_matrix1, annot=False, cmap='coolwarm', xticklabels=keys, yticklabels=keys, ax=ax[0])
#   ax[0].set_title(name1)
#   sb.heatmap(corr_matrix2, annot=False, cmap='coolwarm', xticklabels=keys, yticklabels=keys, ax=ax[1])
#   ax[1].set_title(name2)
#   plt.savefig(f"figures/correlation_matrices/{dset}/{tech}/{dset}_{tech}_corr_{name1}_{name2}.png")
  

import seaborn as sns
import warnings
def plot_two_correlation_matrices(corr_matrix1, corr_matrix2, keys, name1, name2):
  # plot the heatmaps of the correlation matrix on two axes and only have a colorbar for the right one
  fig, ax = plt.subplots(1, 2, figsize=(20, 10))

  fig, axn = plt.subplots(1, 2,  figsize=(20, 10), sharex=True, sharey=True)
  cbar_ax = fig.add_axes([.91, .3, .03, .4])
  fontsize = 35
  
  for i, ax in enumerate(axn.flat):
      sns.heatmap(corr_matrix1 if i == 0 else corr_matrix2,
                  ax=ax,
                  cbar=i == 0,
                  # vmin=0, vmax=1,
                  cbar_ax=None if i else cbar_ax)
      ax.set_title(name1 if i == 0 else name2, fontsize=fontsize+5, fontweight="bold")
      ax.set_xticklabels(keys, rotation=90, fontsize=fontsize, fontweight="bold")
      ax.set_yticklabels(keys, rotation=0, fontsize=fontsize, fontweight="bold")
      if i == 0:
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=fontsize)
  fig.tight_layout(rect=[0, 0, .9, 1])

  # fig.tight_layout(rect=[0, 0, .9, 1])

  # sb.heatmap(corr_matrix1, annot=False, cmap='coolwarm', xticklabels=keys, yticklabels=keys, ax=ax[0])
  
  # sb.heatmap(corr_matrix2, annot=False, cmap='coolwarm', xticklabels=keys, yticklabels=keys, ax=ax[1])
  # ax[1].set_title(name2)
  plt.savefig(f"figures/correlation_matrices/{dset}/{tech}/{dset}_{tech}_corr_{name1}_{name2}.png")
  plt.close()


noq_corr, noq_corr_b1, noq_corr_b2, noq_corr_fc, _ = None, None, None, None, None
for i in range(len(model_files)):
  all_correlations_dic, corr_matrix, corr_matrix_b1, corr_matrix_b2, corr_matrix_fc, all_keys = calculate_correlation_matrices(model_files[i],
                                                                  step=90)
  if "noq" in model_files[i]:
    noq_corr = corr_matrix
    noq_corr_b1 = corr_matrix_b1
    noq_corr_b2 = corr_matrix_b2
    noq_corr_fc = corr_matrix_fc
  else:
    # dsads bic noq  ours 16  ours 8  luq 16  luq 8 
# dsads icarl noq  ours 16  ours 8 luq 16  luq 8 
    # icarl
    # 80.35/9.82  noq
    # HDQT (Ours) 16 Bits diff to noq:              1.4909   1.2118    1.4357
    # 78.86/20.61 LUQ 16 Bits diff to noq:          1.0728   0.6021    0.2358
    # HDQT (Ours) 8 Bits diff to noq:               2.3377   3.2205    3.1627
    # 52.72/44.39 LUQ 8 Bits diff to noq:           0.0451   6.6810    3.9142
    # bic
    # 80.35/9.82  noq 
    # 82.19/4.39  HDQT (Ours) 16 Bits diff to noq:  0.1536   0.1417   -0.4950
    # 71.14/14.39 LUQ 16 Bits diff to noq:          0.2573   0.1589   -0.7075
    # 70.88/18.51 HDQT (Ours) 8 Bits diff to noq:   1.4517   0.7267    0.3094
    # 33.77/21.49 LUQ 8 Bits diff to noq:           2.7605   4.3456   -0.4998
    print(f"{names[i]} diff to noq: {np.round(np.sum(noq_corr_b1-corr_matrix_b1),3)}   {np.round(np.sum(noq_corr_b2-corr_matrix_b2),3)}   {np.round(np.sum(noq_corr_fc-corr_matrix_fc),3)}")
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      plot_two_correlation_matrices(noq_corr, corr_matrix, list(range(10)), "NoQ", names[i])
  plot_correlation_matrix(corr_matrix, list(range(10)), names[i])
  plot_correlation_matrix(corr_matrix_b1, list(range(10)), names[i] + ' Bl1')
  plot_correlation_matrix(corr_matrix_b2, list(range(10)), names[i] + ' Bl2')
  plot_correlation_matrix(corr_matrix_fc, list(range(10)), names[i] + ' FC')





