import torch
import torch.nn as nn
import torch.nn.functional as F
import backbones.linears


def get_activation(act_fun):
  if act_fun == "relu":
    return nn.ReLU()
  elif act_fun == "tanh":
    return nn.Tanh()
  elif act_fun == "sigmoid":
    return nn.Sigmoid()
  else:
    assert 0


class linl(nn.Module):

  def __init__(self, in_dim, out_dim, act_fun):
    super().__init__()
    self.lw = nn.Linear(in_dim, out_dim)
    self.act = get_activation(act_fun)

  def forward(self, x):
    
    x = self.lw(x)
    
    # TOOD: no if block
    if type(x) is dict:
      x = x['logits']

    return self.act(x)

class FCNet(nn.Module):

  def __init__(self, in_dim, hid_dim, out_dim, nr_hid_layers, act_fun="relu"):
    super().__init__()
    self.out_dim = out_dim
    self.net = self.make_layers(
        in_dim, hid_dim, out_dim, nr_hid_layers, act_fun)

  def forward(self, x):
    features = self.net(x)
    return {"features": features}

  def make_layers(self, in_dim, hid_dim, out_dim, nr_hid_layers, act_fun="relu"): 
    layers = []
    
    layer = linl(in_dim, hid_dim, act_fun)
    layers.append(layer)

    for _ in range(nr_hid_layers):
      layer = linl(hid_dim, hid_dim, act_fun)
      layers.append(layer)
    
    layer = linl(hid_dim, out_dim, act_fun)
    layers.append(layer)

    return nn.Sequential(*layers)

  @property
  def last_layer(self):
    return self.net[-1]


