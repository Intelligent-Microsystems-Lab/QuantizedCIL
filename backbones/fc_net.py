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

  def __init__(self, in_dim, out_dim, act_fun, bias):
    super().__init__()
    self.lw = nn.Linear(in_dim, out_dim, bias=bias)
    self.act = get_activation(act_fun)

  def forward(self, x):
    
    x = self.lw(x)
    
    # TODO: no if block
    if type(x) is dict:
      x = x['logits']

    return self.act(x)

class FCNet(nn.Module):

  def __init__(self, in_dim, hid_dim, out_dim, nr_hid_layers, act_fun="relu",
               bias=False, half_dims=False):
    super().__init__()
    self.out_dim = out_dim
    self.net = self.make_layers(
        in_dim, hid_dim, out_dim, nr_hid_layers, act_fun, bias, half_dims)

  def forward(self, x):
    features = self.net(x)
    return {"features": features}

  def make_layers(self, in_dim, hid_dim, out_dim, nr_hid_layers, act_fun="relu",
                  bias=False): 
    layers = []
    
    layer = linl(in_dim, hid_dim, act_fun, bias)
    layers.append(layer)

    old_hid_dim = None
    for _ in range(nr_hid_layers):
      if half_dims:
        old_hid_dim = hid_dim
        hid_dim = hid_dim//2
        layer = linl(old_hid_dim, hid_dim, act_fun, bias)
      else:
        layer = linl(hid_dim, hid_dim, act_fun, bias)
      layers.append(layer)
    
    if half_dims:
      old_hid_dim = hid_dim
      hid_dim = hid_dim//2
      layer = linl(old_hid_dim, hid_dim, act_fun, bias)
      self.out_dim = hid_dim
    else:
      layer = linl(hid_dim, out_dim, act_fun, bias)
    layers.append(layer)

    return nn.Sequential(*layers)

  @property
  def last_layer(self):
    return self.net[-1]


