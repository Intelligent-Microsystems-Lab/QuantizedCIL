# import jax
# import jax.numpy as jnp
# import flax
# import flax.linen as nn
# from jax.tree_util import Partial, tree_flatten, tree_unflatten, tree_structure, tree_leaves
# from randman_dataset import make_spiking_dataset
# import randman_dataset as rd
# from utils import gen_test_data, cos_sim_train_func, online_sim_train_func, custom_snn, bp_snn

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
# task = 0

zoom = False
if zoom:
  nr_tasks = 1
else:
  nr_tasks = 1

for t in range(nr_tasks):
  resolution = 15
  task_to_display = t
  default_file = 2
  nr_classes = 2*(task_to_display+1) if task_to_display < 9 else 11
  epsilon = 1e-10

  # model_accs = [0.7842,,,,]
  if zoom:
    model_files = ['logs/dsads/icarl/weights/fcnet_noq_accbits_16_1994.npy',
                  'logs/dsads/icarl/weights/fcnet_ours_accbits_16_1994.npy',
                  'logs/dsads/icarl/weights/fcnet_luq_corrected_accbits_16_1994.npy',
                  'logs/dsads/icarl/weights/fcnet_ours_accbits_8_1994.npy',
                  ]
    names = ['NoQ','HDQT (Ours) 16 Bits','LUQ 16 Bits','HDQT (Ours) 8 Bits',
            ]
  else:
    model_files = ['logs/dsads/icarl/weights/fcnet_noq_accbits_16_1994.npy',
                  'logs/dsads/icarl/weights/fcnet_ours_accbits_16_1994.npy',
                  'logs/dsads/icarl/weights/fcnet_luq_corrected_accbits_16_1994.npy',
                  'logs/dsads/icarl/weights/fcnet_ours_accbits_8_1994.npy',
                  'logs/dsads/icarl/weights/fcnet_luq_corrected_accbits_8_1994.npy',
                  ]
    names = ['NoQ','HDQT (Ours) 16 Bits','LUQ 16 Bits','HDQT (Ours) 8 Bits',
            'LUQ 8 Bits',
            ]

  data_manager = DataManager(
        'dsads',
        True,
        1994,
        2,
        2,
    )

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(405, 405, bias = False)
      self.fc2 = nn.Linear(405, 405, bias = False)
      self.fc3 = nn.Linear(405, nr_classes, bias = False)

    def forward(self, x):
        
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      x = F.relu(x)
      # output = F.softmax(x, dim=1)

      return x

  # rot dunkelblau gruen lila orange
  colors = ['#e41a1c',
            '#377eb8','#4daf4a','#984ea3','#ff7f00']
  # marker_colors = [sb.set_hls_values('#e41a1c',l=0.2),
  #                  sb.set_hls_values('#377eb8',l=0.2),
  #                  sb.desaturate('#4daf4a',0.5),
  #                  sb.desaturate('#984ea3',0.5),
  #                  sb.desaturate('#ff7f00',0.5)]

  def load_params(file='def', task=0, step=0):

    if file == 'def':
      data_s = np.load(model_files[default_file], allow_pickle=True)      
    else:
      data_s = np.load(file, allow_pickle=True)  
    data_s = np.reshape(data_s,(-1))[0]

    all_params = []
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
    

    return np.concatenate(all_params), data_s[task][step].keys()

  def project1d(w, d):
    assert len(w) == len(d), "dimension does not match for w and "
    return np.dot(w, d) / np.linalg.norm(d)


  def project2d(d, dx, dy, proj_method):
    if proj_method == "cos":
      # when dx and dy are orthorgonal
      x = project1d(d, dx)
      y = project1d(d, dy)
    elif proj_method == "lstsq":
      A = np.vstack([dx, dy]).T
      [x, y] = np.linalg.lstsq(A, d)[0]

    return x, y


  test_dataset = data_manager.get_dataset(
        np.arange(0, nr_classes), source="test", mode="test"
    )
  test_loader = DataLoader(
      test_dataset, batch_size=128, shuffle=False,
      num_workers=4
  )

  def get_loss(model):

    model.eval()
    correct, total, loss, cnt = 0, 0, [], []
    for i, (_, inputs, targets) in enumerate(test_loader):
      inputs = inputs.cuda()
      targets = targets.cuda()
      with torch.no_grad():
        outputs = model(inputs)# ["logits"]
      predicts = torch.max(outputs, dim=1)[1]
      correct += (predicts == targets).sum()
      total += len(targets)

      loss.append(F.cross_entropy(outputs, targets))
      cnt.append(outputs.shape[0])

    return torch.tensor(loss) @ torch.tensor(cnt, dtype = torch.float) / np.sum(cnt)


  def get_surface(model, x, y, xdirection, ydirection, variables):

    xv, yv = np.meshgrid(x, y)


    zv_list = np.ones((resolution,resolution)) * -1
    for i in range(resolution):
      for j in range(resolution):

        model = load_w(model, variables + xv[i,j] * xdirection + yv[i,j] * ydirection)# interpolate_vars)
        zv_list[i,j] = get_loss(model)
        print('.', end='')

    return xv, yv, np.stack(zv_list).flatten().reshape(xv.shape)


  params_end, _ = load_params('def',task_to_display,90)


  matrix = []
  for i in [0,10,20,30,40,50,60,70,80]:
    tmp, _ = load_params('def',task_to_display,i)
    diff_tmp = tmp - params_end
    matrix.append(diff_tmp) # TODO add all models to matrix 



  pca = PCA(n_components=2)
  pca.fit(np.array(matrix))

  pc1 = np.array(pca.components_[0])
  pc2 = np.array(pca.components_[1])

  angle = np.dot(pc1, pc2) / (np.linalg.norm(pc1) * np.linalg.norm(pc2))

  xdirection = pc1
  ydirection = pc2

  ratio_x = pca.explained_variance_ratio_[0]
  ratio_y = pca.explained_variance_ratio_[1]

  dx = pc1
  dy = pc2

  xcoord = {}
  ycoord = {}
  x_abs_max = 0
  y_abs_max = 0


  for j in range(len(model_files)):
    xcoord[j] = []
    ycoord[j] = []
    for i in [0,10,20,30,40,50,60,70,80,90]:

      tmp, dummy_keys  = load_params(model_files[j], task_to_display, i)
      diff_tmp = tmp - params_end

      tmp_x, tmp_y = project2d(diff_tmp, dx, dy, 'cos')
      xcoord[j].append(tmp_x)
      ycoord[j].append(tmp_y)

      if np.abs(tmp_x) > x_abs_max:
        x_abs_max = abs(tmp_x)
      if np.abs(tmp_y) > y_abs_max:
        y_abs_max = abs(tmp_y)

  buffer_y = y_abs_max * 0.05
  buffer_x = x_abs_max * 0.05

  x = np.linspace(
      (-1*x_abs_max) - buffer_x,
      x_abs_max + buffer_x,
      resolution,
  )
  y = np.linspace(
      (-1*y_abs_max) - buffer_y,
      y_abs_max + buffer_y,
      resolution,
  )



  model = Net().cuda()



  def load_w(model, params):
    with torch.no_grad():
      model.fc1.weight = nn.Parameter(torch.tensor(np.reshape(params[:405*405], (405,405))).cuda())
      model.fc2.weight = nn.Parameter(torch.tensor(np.reshape(params[405*405:405*405*2], (405,405))).cuda())
      model.fc3.weight = nn.Parameter(torch.tensor(np.reshape(params[405*405*2:(405*405*2)+(405*nr_classes)], (nr_classes,405))).cuda())

    return model




  model = load_w(model, params_end)

  xv, yv, zv = get_surface(model, x, y, xdirection, ydirection, params_end)

  # import pdb; pdb.set_trace()
  try:
    zv = np.log(zv+epsilon)
  except:
    import pdb; pdb.set_trace()


  font_size = 23
  gen_lw = 8

  plt.rc("font", weight="bold")
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14.4, 8.5))


  def fmt(x):
    s = f"{x:.3f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

  CS = ax.contour(xv, yv, zv, resolution)
  ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

  for j in range(len(model_files)):
    # noq is 2
    # if j == 2:
    #   continue
    # import pdb; pdb.set_trace()
    ax.plot(
        xcoord[j],
        ycoord[j],
        label=names[j],
        color=colors[j],
        marker="o",
        markeredgecolor='black',
        markerfacecolor="None",
        markersize=8,
        linewidth=gen_lw,
        alpha=0.5,
    )
    ax.plot(
        xcoord[j][-1],
        ycoord[j][-1],
        label="_nolegend_",
        color=colors[j],
        marker="X",
        markeredgecolor='black',
        markerfacecolor=colors[j],
        markersize=8,
        linewidth=gen_lw,
    )
    ax.plot(
        xcoord[j][0],
        ycoord[j][0],
        label="_nolegend_",
        color=colors[j],
        marker="*",
        markeredgecolor='black',
        markerfacecolor=colors[j],
        markersize=8,
        linewidth=gen_lw,
    )

  # ax.legend(['OTTT','Approx OTPE', 'OSTL', 'OTPE', 'BPTT'],fontsize=32)
  ax.set_xlabel(
      "1st PC: %.2f %%" % (ratio_x * 100),
      fontdict={"weight": "bold", "size": font_size+3},
  )
  ax.set_ylabel(
      "2nd PC: %.2f %%" % (ratio_y * 100),
      fontdict={"weight": "bold", "size": font_size+3},
  )
  ax.tick_params(axis="both", which="major", labelsize=font_size/2)

  if task_to_display == 0 and not zoom:
    ax.legend(
      # names,
      # # ["OTTT", "Approx OTPE", "OSTL"],
      # #loc="upper left",
      fontsize=font_size+3,
      # frameon=False,
    )
  plt.tight_layout()
  if zoom:
    plt.savefig(f"figures/loss_landscapes/ll_task_{task_to_display}_zoom.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/loss_landscapes/ll_task_{task_to_display}_zoom.svg", dpi=300, bbox_inches="tight")
  else:
    plt.savefig(f"figures/loss_landscapes/ll_task_{task_to_display}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/loss_landscapes/ll_task_{task_to_display}.svg", dpi=300, bbox_inches="tight")
  plt.close()
