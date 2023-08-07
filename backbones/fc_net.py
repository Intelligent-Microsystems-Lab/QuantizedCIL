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

class FCNet(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, nr_hid_layers, act_fun="relu"):
        super().__init__()
        self.out_dim = out_dim
        self.net = self.make_layers(in_dim, hid_dim, out_dim, nr_hid_layers, act_fun)

    def forward(self, x):
        features = self.net(x)
        return {"features": features}
        
    def make_layers(self, in_dim, hid_dim, out_dim, nr_hid_layers, act_fun="relu"): 
        layers = []
        layers.append(nn.Linear(in_dim, hid_dim))
        layers.append(get_activation(act_fun))
        for _ in range(nr_hid_layers):
            layers.append(nn.Linear(hid_dim, hid_dim))
            layers.append(get_activation(act_fun))
        layers.append(nn.Linear(hid_dim, out_dim))
        return nn.Sequential(*layers)
    
    @property
    def last_layer(self):
        return self.net[-1]

    
