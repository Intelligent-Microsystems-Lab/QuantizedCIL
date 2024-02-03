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
task = 0

resolution = 10


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
    self.fc3 = nn.Linear(405, 11, bias = False)

  def forward(self, x):
      
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.relu(x)
    # output = F.softmax(x, dim=1)

    return x

# model = factory.get_model('FC', args)

# output_size = 10
# nlayers = 3
# dim = 3
# seq_len = 50
# lr = "001"
# manifold_seed_val = 0
# init_seed_val = 0
# manifold_seed = jax.random.PRNGKey(manifold_seed_val)
# init_seed = jax.random.split(jax.random.PRNGKey(init_seed_val))[0]
# dtype = jnp.float32
# slope = 25
# tau = dtype(2.)
# batch_sz = 128
# spike_fn = sl.fs(slope)
# n_iter = 20000
# layer_name = 128
# update_time = 'offline'
# timing = 'rate'
# if timing=='time':
#     t_name = 'time'
#     t = True
# elif timing == 'rate':
#     t_name = 'rate'
#     t = False
# if layer_name == 128:
#     layer_sz = lambda i: 128
# elif layer_name == 512:
#     layer_sz = lambda i: 512
# elif layer_name == 256:
#     layer_sz = lambda i: 256

# #t = True
# output_size = 10
# dtype = jnp.float32
# seed = 0
# key = jax.random.PRNGKey(seed)
# key2 = jax.random.split(key,num=1)[0]
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
marker_colors = [sb.set_hls_values('#e41a1c',l=0.2),
                 sb.set_hls_values('#377eb8',l=0.2),
                 sb.desaturate('#4daf4a',0.5),
                 sb.desaturate('#984ea3',0.5),
                 sb.desaturate('#ff7f00',0.5)]

# model_indicator = 4

# gen_data = Partial(rd.make_spiking_dataset,nb_classes=10, nb_units=50, nb_steps=seq_len, nb_samples=1000, dim_manifold=dim, alpha=1., nb_spikes=1, seed=manifold_seed,shuffle=True,time_encode=t,dtype=dtype)



# class bp_mlp_variable(nn.Module):
#   n_layers: int = 5
#   sz: int = 128

#   def setup(self):
#     snns = list()
#     snns.append(sl.SpikingBlock(nn.Dense(10), sl.subLIF(tau, spike_fn)))
#     for i in range(1, self.n_layers):
#       snns.append(sl.SpikingBlock(nn.Dense(self.sz), sl.subLIF(tau, spike_fn)))
#     self.snns = snns

#   def __call__(self, carry, s):
#     if carry[0]['u'].size == 0:
#       in_ax1 = None
#     else:
#       in_ax1 = 0

#     for i in range(self.n_layers):
#       carry[i]['u'], s = jax.vmap(
#           self.snns[self.n_layers - (i + 1)], in_axes=(in_ax1, 0))(carry[i]['u'], s)

#     return carry, s


# def load_params(file):
#   with open(file, 'rb') as f:
#     all_params = pickle.load(f)
#   struct = tree_structure(all_params[-1])
#   for i in range(len(all_params) - 1):
#     all_params[i] = tree_unflatten(struct, tree_leaves(all_params[i]))
#   return all_params


def load_params(file, task=0, step=0):

  data_s = np.load('logs/dsads/icarl/weights/fcnet_noq_accbits_8_1994.npy', allow_pickle=True)      
  data_s = np.reshape(data_s,(-1))[0]

  all_params = []
  for k,v in data_s[task][step].items():
    # print(k)
    all_params.append(np.array(v.flatten().cpu().numpy()))

  return np.concatenate(all_params)

# def npvec_to_tensorlist(pc, params):
#   tree_val, tree_struct = jax.tree_util.tree_flatten(params)
#   val_list = []
#   counter = 0
#   for i in [x.shape for x in tree_val]:
#     increase = np.prod(i)
#     val_list.append(pc[counter: int(counter + increase)].reshape(i))
#     counter += increase

#   return jax.tree_util.tree_unflatten(tree_struct, val_list)

# def npvec_to_tensorlist(pc, params):



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


# model = bp_snn(output_sz=output_size, n_layers=nlayers, spike_fn=spike_fn, layer_sz=layer_sz, dtype=dtype)

# carry = [{'u': jnp.zeros((batch_sz, 10))}] + \
#     [{'u': jnp.zeros((batch_sz, layer_name))}] * (nlayers - 1)
# carry = carry[::-1]


# def loss_fn(params, carry, b):
#   c, s = model.apply(params, carry, b[0])
#   loss = jnp.mean(optax.softmax_cross_entropy(s, b[1]))
#   return c, loss


test_dataset = data_manager.get_dataset(
      np.arange(0, 11), source="test", mode="test"
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


  # data, logits = gen_data(seed2=gen_key)

  # p_loss = Partial(loss_fn, params)
  # c, loss = jax.lax.scan(
  #     p_loss, carry, (data[:, :batch_sz], logits[:, :batch_sz]))
  # return jnp.mean(loss)


def get_surface(model, x, y, xdirection, ydirection, variables):

  xv, yv = np.meshgrid(x, y)

  # def surface_parallel(ix, iy):


  #   interpolate_vars = jax.tree_util.tree_map(
  #       lambda w, x, y: w + x * ix + y * iy,
  #       variables,
  #       xdirection,
  #       ydirection,
  #   )

  #   model = load_w(model, params_end)

  #   return get_loss(interpolate_vars, key2)


  zv_list = np.ones((resolution,resolution)) * -1
  for i in range(resolution):
    for j in range(resolution):
      # zv = torch.vmap(surface_parallel)(
      #     jnp.array(xv.flatten())[(i * 100): (i + 1) * 100],
      #     jnp.array(yv.flatten())[(i * 100): (i + 1) * 100],
      # )

      # import pdb; pdb.set_trace()
      # interpolate_vars = jax.tree_util.tree_map(
      #     lambda w, x, y: w + x * ix + y * iy,
      #     variables,
      #     xdirection,
      #     ydirection,
      # )

      model = load_w(model, variables + xv[i,j] * xdirection + yv[i,j] * ydirection)# interpolate_vars)
      zv_list[i,j] = get_loss(model)
      print('.', end='')

      # zv_list.append(get_loss(model))

  return xv, yv, np.stack(zv_list).flatten().reshape(xv.shape)


params_end = load_params('bla',9,90)

# load_params(
# 'randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,n_iter,manifold_seed_val,t_name,slope,lr,init_seed_val))
    #'data_final/models/model_{}layer_{}_3dim_20seqlen_{}iter_1sp_{}seed_time_sub_adamax'

matrix = []
for i in [0,10,20,30,40,50,60,70,80]:
  tmp  = load_params('bla',9,i)
  diff_tmp = tmp - params_end
  matrix.append(diff_tmp)
# for i in range(0,n_iter+200,200):
#   for j in range(5):
#     tmp = load_params(
# 'randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,i,manifold_seed_val,t_name,slope,lr,init_seed_val))
#     diff_tmp = jax.tree_map(lambda x, y: x - y, tmp[j], params_end[model_indicator])
#     matrix.append(jnp.hstack([x.reshape(-1)
#                   for x in jax.tree_util.tree_flatten(diff_tmp)[0]]))


pca = PCA(n_components=2)
pca.fit(np.array(matrix))

pc1 = np.array(pca.components_[0])
pc2 = np.array(pca.components_[1])

angle = np.dot(pc1, pc2) / (np.linalg.norm(pc1) * np.linalg.norm(pc2))

xdirection = pc1 # npvec_to_tensorlist(pc1, params_end)
ydirection = pc2 # npvec_to_tensorlist(pc2, params_end)

ratio_x = pca.explained_variance_ratio_[0]
ratio_y = pca.explained_variance_ratio_[1]

dx = pc1
dy = pc2

xcoord = {}
ycoord = {}
x_abs_max = 0
y_abs_max = 0


for j in range(1):
  xcoord[j] = []
  ycoord[j] = []
  for i in [0,10,20,30,40,50,60,70,80,90]:

    tmp  = load_params('bla',9,i)
    diff_tmp = tmp - params_end
    #     tmp = load_params(
    # 'randman_data/models/model_{}layer_{}_{}dim_{}_{}seqlen_{}iter_{}manifold_{}_sub_{}fs_adamax_lr{}_{}seed'.format(nlayers,layer_name,dim,update_time,seq_len,i,manifold_seed_val,t_name,slope,lr,init_seed_val))
    #     diff_tmp = jax.tree_map(lambda x, y: x - y, tmp[j], params_end[model_indicator])
    #     diff_tmp = jnp.hstack([x.reshape(-1)
    #                           for x in jax.tree_util.tree_flatten(diff_tmp)[0]])

    tmp_x, tmp_y = project2d(diff_tmp, dx, dy, 'cos')
    xcoord[j].append(tmp_x)
    ycoord[j].append(tmp_y)

    if np.abs(tmp_x) > x_abs_max:
      x_abs_max = abs(tmp_x)
    if np.abs(tmp_y) > y_abs_max:
      y_abs_max = abs(tmp_y)




# buffer_y = (np.max(ycoord) - np.min(ycoord)) * 0.05
# buffer_x = (np.max(xcoord) - np.min(xcoord)) * 0.05

# x = np.linspace(
#     np.min(xcoord) - buffer_x,
#     np.max(xcoord) + buffer_x,
#     100,
# )
# y = np.linspace(
#     np.min(ycoord) - buffer_y,
#     np.max(ycoord) + buffer_y,
#     100,
# )


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
    model.fc3.weight = nn.Parameter(torch.tensor(np.reshape(params[405*405*2:(405*405*2)+(405*11)], (11,405))).cuda())

  return model




model = load_w(model, params_end)

xv, yv, zv = get_surface(model, x, y, xdirection, ydirection, params_end) # params_end[model_indicator])

# import pdb; pdb.set_trace() # martin hier....
font_size = 23
gen_lw = 8

plt.rc("font", weight="bold")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14.4, 8.5))


def fmt(x):
  s = f"{x:.3f}"
  if s.endswith("0"):
    s = f"{x:.0f}"
  return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"


#CS = ax.contourf(xv, yv, zv, 100)
CS = ax.contour(xv, yv, zv, resolution)
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)

for j in range(1):
  ax.plot(
      xcoord[j],
      ycoord[j],
      label=str(j),
      color=colors[j],
      marker="o",
      markeredgecolor='black',
      markerfacecolor="None",
      markersize=8,
      linewidth=gen_lw,
  )

ax.legend(['OTTT','Approx OTPE', 'OSTL', 'OTPE', 'BPTT'],fontsize=32)
ax.set_xlabel(
    "1st PC: %.2f %%" % (ratio_x * 100),
    fontdict={"weight": "bold", "size": font_size},
)
ax.set_ylabel(
    "2nd PC: %.2f %%" % (ratio_y * 100),
    fontdict={"weight": "bold", "size": font_size},
)

plt.tight_layout()
plt.savefig("figures/ll_test.svg", dpi=300, bbox_inches="tight")
plt.close()
