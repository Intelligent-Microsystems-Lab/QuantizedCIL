# IMSL Lab - University of Notre Dame | University of St Andrews
# Author: Clemens JS Schaefer | Martin Schiemer
# Quantized training.

import scipy
import numpy as np
import functools
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import  transforms

from torch.nn.parameter import Parameter
from torch.autograd.function import Function
from torch.autograd.function import InplaceFunction

from backbones.linears import SimpleLinear

from utils.data_manager import DummyDataset
from torch.utils.data import DataLoader

from squant_function import SQuant_func

from hadamard import make_hadamard, biggest_power2_factor


track_stats = {'grads': {}, 'acts': {}, 'wgts': {},
               'grad_stats': {}, 'test_acc': [], 'train_acc': [], 'loss': [],
               'zeros': {}, 'maxv':{}}
calibrate_phase = False
quantizeFwd = False
quantizeBwd = False
quantCalibrate = "max"
quantTrack = False
quantBits = 4
quantAccBits = 8
quantWgtStoreBits = 8
quantMethod = 'ours'
quantBatchSize = 128
quantFWDWgt = 'int'
quantFWDAct = 'int'
quantBWDWgt = 'int'
quantBWDAct = 'int'
quantBWDGrad1 = "int"
quantBWDGrad2 = "int"
quantBlockSize = 32
quantRelevantMeasurePass = False
quantUpdateScalePhase = False
quantUpdateLowThr = .7
quantUpdateHighThr = .3
global_args = None
quant_no_update_perc = None

current_uname = ''

QpW = None
QnW = None
QpA = None
QnA = None

quantGradMxScale = 1.

scale_library = {}


class QuantMomentumOptimizer(torch.optim.Optimizer):

  # Init Method:
  def __init__(self, params, lr=1e-3, momentum=0.9):
    super(QuantMomentumOptimizer, self).__init__(params, defaults={'lr': lr})
    self.momentum = momentum
    self.state = dict()
    self.layer_lr = dict()
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is not None:
          self.state[p] = dict(mom=torch.zeros_like(p.data))
          self.layer_lr[p] = self.param_groups[0]['lr']

  # Step Method
  def step(self):
    # print('------')
    for group in self.param_groups:
      for p in group['params']:
        if p not in self.state:
          self.state[p] = dict(mom=torch.zeros_like(p.data))
          self.layer_lr[p] = self.param_groups[0]['lr']
        if p.grad is not None:

          # backup = copy.deepcopy(p.data)

          # if (self.layer_lr[p] * p.grad.data).max() < 1.0:
          #   self.layer_lr[p] *= 1.1
          #   # p.data -=  (torch.bernoulli(torch.ones_like(p.data) * .5) * 2 - 1 ) * torch.bernoulli(torch.ones_like(p.data) * .5)
          # else:
          p.data -= self.layer_lr[p] * p.grad.data

          p.data = torch.clip(
              p.data, -(2**(quantWgtStoreBits - 1) - 1), (2**(quantWgtStoreBits - 1) - 1))
          p.data = torch.round(p.data)

          # import pdb; pdb.set_trace()

          # if quantUpdateScalePhase:
          #   scale_library[current_uname] = (int(torch.sum(output == 0.))/np.prod(output.shape),
          #                               max(int(torch.sum(output == n))/np.prod(output.shape),
          #                                   int(torch.sum(output == -n))/np.prod(output.shape)))
          # np.mean((backup == p.data).cpu().numpy())
          # backup == p.data
          # print(torch.abs(backup - p.data).max())
          # import pdb; pdb.set_trace()


def init_properties(obj, uname):
  obj.fullName = ''
  obj.statistics = []
  obj.layerIdx = 0

  # obj.alpha = Parameter(torch.tensor([1], dtype=torch.float32))
  # obj.beta = Parameter(torch.tensor([1], dtype=torch.float32))
  obj.abits = quantBits
  obj.wbits = quantBits

  obj.QnW = -2 ** (obj.wbits - 1) + 1
  obj.QpW = 2 ** (obj.wbits - 1) - 1
  obj.QnA = 0
  obj.QpA = 2 ** obj.abits - 1

  global QpW
  global QnW
  global QpA
  global QnA
  QpW = obj.QpW
  QnW = obj.QnW
  QpA = obj.QpA
  QnA = obj.QnA

  obj.quantizeFwd = quantizeFwd
  obj.quantizeBwd = quantizeBwd

  obj.c1 = 12.1
  obj.c2 = 12.2
  obj.stochastic = quantBWDGrad1
  obj.calibrate = quantCalibrate
  obj.repeatBwd = 1

  obj.uname = uname




def place_quant(m, lin_w, lin_b, c_path='',):

  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      if isinstance(target_attr, nn.Conv2d):
        if not hasattr(target_attr, 'c1'):
          if quantMethod == 'luq':
            tmp_meth = Conv2d_LUQ
          elif quantMethod == 'ours':
            tmp_meth = Conv2d_Ours
          else:
            raise Exception('Unknown quant method: ' + quantMethod)
          setattr(m, attr_str,
                  tmp_meth(in_channels=target_attr.in_channels,
                           out_channels=target_attr.out_channels,
                           kernel_size=target_attr.kernel_size,
                           stride=target_attr.stride,
                           padding=target_attr.padding,
                           padding_mode=target_attr.padding_mode,
                           dilation=target_attr.dilation,
                           groups=target_attr.groups,
                           bias=hasattr(target_attr, 'bias'),
                           uname=c_path + '_' + attr_str,))
      if isinstance(target_attr, nn.Linear) or isinstance(target_attr,
                                                          SimpleLinear):
        if not hasattr(target_attr, 'c1'):
          if quantMethod == 'luq':
            tmp_meth = Linear_LUQ
          elif quantMethod == 'ours':
            tmp_meth = Linear_Ours
          else:
            raise Exception('Unknown quant method: ' + quantMethod)

          if hasattr(getattr(m, attr_str), 'sw'):
            old_sw = getattr(m, attr_str).sw
          else:
            old_sw = None
          setattr(m, attr_str, tmp_meth(in_features=target_attr.in_features,
                                        out_features=target_attr.out_features,
                                        bias=getattr(
                                            target_attr, 'bias') is not None,
                                        uname=c_path + '_' + attr_str,))

          if lin_w is not None and attr_str == 'fc':
            if quantFWDWgt == 'mem':
              if old_sw is None:
                pass
              else:
                scale_dyn_range = global_args["dyn_scale"]
                # TODO possibly critical!
                # lin_w = lin_w / scale_dyn_range
                # lin_w = torch.round(lin_w)
                # m.fc.sw.data = old_sw * scale_dyn_range
                m.fc.weight.data = lin_w
            else:
              m.fc.weight.data = lin_w
          if lin_b is not None and attr_str == 'fc':
            m.fc.bias = nn.Parameter(lin_b)
  for n, ch in m.named_children():
    place_quant(ch, lin_w, lin_b, c_path + '_' + n,)


def balanced_scale_calibration_fwd(memory_tuple, train_set_copy, known_cl,
                                   total_cl, model, device, data_manager):
  
  global quantUpdateScalePhase
  quantUpdateScalePhase = True
  with torch.no_grad():
    mem_samples, mem_targets = memory_tuple
    samples_per_cl = mem_samples.shape[0] / len(np.unique(mem_targets))
    # get as many samples from the new classes as for each in memory
    train_loader_copy = DataLoader(
        train_set_copy, batch_size=1, shuffle=True
        )
    new_samples = {a:[] for a in range(known_cl, total_cl)}
    for _, input, target in train_loader_copy:
      if len(new_samples[target.item()]) < samples_per_cl:
        new_samples[target.item()].append(input)
        if sum([len(new_samples[key])==samples_per_cl for key in new_samples]) == len(list(new_samples.keys())):
          break

    for cl in range(known_cl, total_cl):
      mem_targets = np.concatenate([mem_targets, torch.tensor([cl] * new_samples[cl].__len__() )])

    # import pdb; pdb.set_trace()
    new_samples = np.concatenate([np.concatenate(new_samples[key], axis=0) for key in new_samples.keys()], axis=0)
    # import pdb; pdb.set_trace()

    try:
      mem_samples = np.concatenate([mem_samples, new_samples], axis=0)
    except:
      # import pdb; pdb.set_trace()
      mem_samples = np.reshape(mem_samples, (mem_samples.shape[0], new_samples.shape[1],))
      mem_samples = np.concatenate([mem_samples, new_samples], axis=0)

    # TODO: make batchsize equal to args batch_size and sample stratified
    # TODO: datatype HAR wrong @ms400?
    # try:
    # import pdb; pdb.set_trace()
    # try:
    update_loader = DataLoader(
        DummyDataset(torch.tensor(mem_samples), torch.tensor(mem_targets), transforms.Compose([*data_manager._train_trsf,]), datatype = 'HAR' if len(mem_samples.shape) <= 2 else 'image'),
        batch_size=len(mem_samples), shuffle=True
        )

    for _, inputs, targets in update_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      model(inputs)
      break
    del train_set_copy, train_loader_copy, update_loader
  
  update_scale(model)
  quantUpdateScalePhase = False


def update_scale(m, c_path='',):

  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      if isinstance(target_attr, Conv2d_Ours):
        if hasattr(target_attr, 'c1'):
          c_name = c_path + '_' + attr_str
          if scale_library[c_path + '_' + attr_str][1] > quantUpdateHighThr:
            with torch.no_grad():
              target_attr.weight /= 2
              # TODO: maybe try ceil
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              # TODO: check if times two is correct
              target_attr.sw *= 2
              # print('increased scale '+c_path + '_' + attr_str)
            # elif quant_no_update_perc[c_name.replace('_', '.')[1:] + '.weight'] > quantUpdateLowThr:
          elif scale_library[c_path + '_' + attr_str][1] < quantUpdateLowThr:
            with torch.no_grad():
              target_attr.weight *= 2
              # TODO: maybe try ceil
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              # TODO: check if times two is correct
              target_attr.sw /= 2
              # print('decreased scale '+c_path + '_' + attr_str)
          
      if isinstance(target_attr, Linear_Ours): # or isinstance(target_attr,
        if hasattr(target_attr, 'c1'):
          # print(attr_str)
          # import pdb; pdb.set_trace()
          c_name = c_path + '_' + attr_str
          if scale_library[c_path + '_' + attr_str][1] > quantUpdateHighThr:
            with torch.no_grad():
              target_attr.weight /= 2
              # TODO: maybe try ceil
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              # TODO: check if times two is correct
              target_attr.sw *= 2
              # print('increased scale '+c_path + '_' + attr_str)
            # elif quant_no_update_perc[c_name.replace('_', '.')[1:] + '.weight'] > quantUpdateLowThr:
          elif  scale_library[c_path + '_' + attr_str][1] < quantUpdateLowThr:
            with torch.no_grad():
              target_attr.weight *= 2
              # TODO: maybe try ceil
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              # TODO: check if times two is correct
              target_attr.sw /= 2
              # print('decreased scale '+c_path + '_' + attr_str)

          
  for n, ch in m.named_children():
    update_scale(ch, c_path + '_' + n,)



def save_lin_params(m):
  weights = None
  bias = None
  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      if isinstance(target_attr, SimpleLinear): # or isinstance(target_attr, Linear_LUQ):
        weights = target_attr.weight
        bias = target_attr.bias
  for n, ch in m.named_children():
    save_lin_params(ch)

  return weights, bias


def dynamic_stoch(x, scale=1):
  # can only be used in BWD - not differentiable
  dim = x.shape
  x = x.flatten()

  mx = x.abs().max() * scale  # torch.quantile(x.abs(), .99) # TODO optimize calibration
  scale = mx / (2**(quantBits - 1) - 1)
  x = torch.clamp(x, -mx, mx) / (scale + 1e-32)

  sign = torch.sign(x)
  xq_ord = torch.floor(x.abs())

  qe = x.abs() - xq_ord

  flip = torch.bernoulli(qe)

  xq = xq_ord + flip

  xq = torch.reshape(sign * xq, dim)
  return xq, scale


def dynamic_intQ(x, scale=None):
  # can only be used in BWD - not differentiable
  # torch.quantile(x.abs(), .99) # TODO optimize calibration
  if scale is None:
    scale = global_args["quantile"]
  mx = x.abs().max() * scale
  scale = mx / (2**(quantBits - 1) - 1)
  x = torch.clamp(x, -mx, mx)
  # epsilion for vmap # TODO eps size?
  return torch.round(x / (scale + 1e-32)), scale

class dynamic_intQ_FWD(Function):

  @staticmethod
  def forward(ctx, x):
    # torch.quantile(x.abs(), .99) # TODO optimize calibration
    mx = x.abs().max() * global_args["quantile"]
    ctx.mx = mx
    ctx.save_for_backward(x)
    if x.min() < 0:
      scale = mx / (2**(quantBits - 1) - 1)
      x = torch.clamp(x, -mx, mx)
    else:
      scale = mx / (2**(quantBits) - 1)
      x = torch.clamp(x, 0, mx)

    return torch.round(x / (scale + 1e-32) ), scale

  @staticmethod
  def backward(ctx, grad_output, grad_scale):
    # STE
    x, = ctx.saved_tensors

    # TODO local scale reconsideration
    # local_mx = ctx.mx
    local_mx = (2**(quantBits - 1) - 1)

    grad_output = torch.where(x > local_mx, torch.tensor(
        [0], dtype=x.dtype, device=x.device), grad_output)
    grad_output = torch.where(
        x < -local_mx, torch.tensor([0], dtype=x.dtype, device=x.device), grad_output)
    # import pdb; pdb.set_trace()
    if torch.isnan(grad_output).any():
      import pdb; pdb.set_trace()
    return grad_output, None



class FLinearQ(torch.autograd.Function):
  generate_vmap_rule = True

  @staticmethod
  def forward(x, w, h_out, h_bs, sx, sw):

    if quantFWDWgt == 'mem':
      # only get quantFWDWgt
      mx = 2**(quantWgtStoreBits - 1) - 1
      scale = mx / (2**(quantBits - 1) - 1)
      w = torch.clamp(w, -mx, mx)
      w = torch.round(w / (scale + 1e-32))


    global current_uname
    # fin_output = torch.zeros((x.shape[0], w.shape[0])).to(x.device)

    fin_output = 0 * F.linear(x[:,0:quantBlockSize], w[:,0:quantBlockSize])

    for i in range(int(np.ceil( x.shape[1]/quantBlockSize ))):
      output = F.linear(x[:,i*quantBlockSize:(i+1)*quantBlockSize], w[:,i*quantBlockSize:(i+1)*quantBlockSize])
      # requantize to acc BW (clamp to big values - no scale)
      n = 2**quantAccBits / 2 - 1
      output = torch.clamp(output, -n, n)


      if quantUpdateScalePhase and i == 0:
        global scale_library
        
        scale_library[current_uname] = (int(torch.sum(output == 0.))/np.prod(output.shape),
                                        max(int(torch.sum(output == n))/np.prod(output.shape),
                                            int(torch.sum(output == -n))/np.prod(output.shape)))

      if quantRelevantMeasurePass and i == 0:
        # import pdb; pdb.set_trace()
        if current_uname in track_stats['zeros']:
          track_stats['zeros'][current_uname].append(torch.sum(output == 0.)/np.prod(output.shape))
        else:
          track_stats['zeros'][current_uname] = [torch.sum(output == 0.)/np.prod(output.shape)]

        if current_uname in track_stats['maxv']:
          track_stats['maxv'][current_uname].append(torch.max(torch.sum(output == n)/np.prod(output.shape), torch.sum(output == -n)/np.prod(output.shape)))
        else:
          track_stats['maxv'][current_uname] = [torch.max(torch.sum(output == n)/np.prod(output.shape) , torch.sum(output == -n)/np.prod(output.shape))]

      fin_output += output

    # fin_output = F.linear(x, w)
    # n = 2**quantAccBits / 2 - 1
    # fin_output = torch.clamp(fin_output, -n, n)


    # global current_uname
    # if current_uname in track_stats['zeros']:
    #   track_stats['zeros'][current_uname].append(int(torch.sum(fin_output == 0.))/np.prod(fin_output.shape))
    # else:
    #   track_stats['zeros'][current_uname] = [int(torch.sum(fin_output == 0.))/np.prod(fin_output.shape)]

    # if current_uname in track_stats['maxv']:
    #   track_stats['maxv'][current_uname].append(max(int(torch.sum(fin_output == n))/np.prod(fin_output.shape), int(torch.sum(fin_output == -n))/np.prod(fin_output.shape)))
    # else:
    #   track_stats['maxv'][current_uname] = [max(int(torch.sum(fin_output == n))/np.prod(fin_output.shape) , int(torch.sum(fin_output == -n))/np.prod(fin_output.shape))]

    # if quantUpdateScalePhase:
    #   global scale_library
    #   # TODO: do not forget to comment back in!!!!
    #   # global current_uname
    #   scale_library[current_uname] = (int(torch.sum(fin_output == 0.))/np.prod(fin_output.shape),
    #                                     max(int(torch.sum(fin_output == n))/np.prod(fin_output.shape),
    #                                         int(torch.sum(fin_output == -n))/np.prod(fin_output.shape)))

    return fin_output * sw * sx

  @staticmethod
  def setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1],
                          inputs[2], inputs[3], inputs[4], inputs[5])

  @staticmethod
  def backward(ctx, grad_output):

    # TODO scales
    x, w, h_out, h_bs, sx, sw = ctx.saved_tensors

    w_h1 = h_out @ w
    # requantize weights
    if quantBWDWgt == 'int':
      w_h1, swh1 = dynamic_intQ(w_h1, scale=1.)
    elif quantBWDWgt == 'noq':
      w_h1 = w_h1
      swh1 = torch.tensor([1.0])
    else:
      raise Exception('Grad rounding scheme not implemented: ' + quantBWDWgt)

    grad_output_h1 = grad_output @ h_out
    # quant grad_output
    if quantBWDGrad1 == 'int':
      grad_output_h1, sg1 = dynamic_intQ(grad_output_h1)
    elif quantBWDGrad1 == 'sq':
      raise Exception('not implemented')
      grad_output_h1 = dynamic_squant(grad_output_h1)
    elif quantBWDGrad1 == 'stoch':
      grad_output_h1, sg1 = dynamic_stoch(grad_output_h1)
    elif quantBWDGrad1 == 'noq':
      grad_output_h1 = grad_output_h1
      sg1 = torch.tensor([1.0])
    else:
      raise Exception('Grad rounding scheme not implemented: ' + quantBWDGrad1)

    # print(torch.unique(grad_output_h1).shape[0])
    # TODO biggest power of two can be optimized
    grad_input = (grad_output_h1 @ w_h1) 

    # if quantPrintStats:
    #   print(grad_input.max())

    n = 2**quantAccBits / 2 - 1
    grad_input = torch.clamp(grad_input, -n, n)

    x_h2 = h_bs @ x
    # requantize acts
    if quantBWDAct == 'int':
      x_h2, sxh2 = dynamic_intQ(x_h2, scale=1.)
    elif quantBWDAct == 'noq':
      x_h2 = x_h2
      sxh2 = torch.tensor([1.0])
    elif quantBWDAct == 'stoch':
      x_h2, sxh2 = dynamic_stoch(x_h2)
    else:
      raise Exception('Grad rounding scheme not implemented: ' + quantBWDAct)

    grad_output_h2 = grad_output.T @ h_bs
    # quant grad_output
    if quantBWDGrad2 == 'int':
      grad_output_h2, sg2 = dynamic_intQ(grad_output_h2)
    elif quantBWDGrad2 == 'sq':
      raise Exception('not implemented')
      grad_output_h2 = dynamic_squant(grad_output_h2)
    elif quantBWDGrad2 == 'stoch':
      grad_output_h2, sg2 = dynamic_stoch(grad_output_h2)
    elif quantBWDGrad2 == 'noq':
      grad_output_h2 = grad_output_h2
      sg2 = torch.tensor([1.0])
    else:
      raise Exception('Grad rounding scheme not implemented: ' + quantBWDGrad2)

    # print(torch.unique(grad_output_h2).shape[0])

    grad_w = (grad_output_h2 @ x_h2) 

    # if quantPrintStats:
    #   print(grad_w.max())

    n = 2**quantAccBits / 2 - 1
    grad_w = torch.clamp(grad_w, -n, n)

    if torch.isnan(grad_input).any():
      import pdb; pdb.set_trace()
    if torch.isnan(grad_w).any():
      import pdb; pdb.set_trace()

    return grad_input * sg1 * swh1 * sw * 1 / biggest_power2_factor(h_out.shape[0]), grad_w * sg2 * sxh2 * sx * 1 / biggest_power2_factor(h_bs.shape[0]), None, None, None, None



class Linear_Ours(nn.Linear):

  """docstring for Conv2d_BF16."""

  def __init__(self, uname, *args, **kwargs):
    super(Linear_Ours, self).__init__(*args, **kwargs)
    init_properties(self, uname)

    self.register_buffer('hadamard_out', torch.tensor(
        make_hadamard(self.out_features), dtype=self.weight.dtype))
    self.register_buffer('hadamard_bs', torch.tensor(
        make_hadamard(quantBatchSize), dtype=self.weight.dtype))

    # self.lsq_act = Parameter(torch.tensor([1.], dtype=torch.float32))
    # self.lsq_wgt = Parameter(torch.tensor(
    #     [self.weight.abs().mean() * 2 / np.sqrt(self.QpW)], dtype=torch.float32))

    if quantFWDWgt == 'mem':
      # if 'backbone' not in uname:
      #   scale_dyn_range = global_args["init_dyn_scale"]
      # else:
      scale_dyn_range = global_args["dyn_scale"]

      # DEBUG!!!!
      # scale_dyn_range = .975
      # DEBUG!!!!

      with torch.no_grad():

        mx = self.weight.abs().max() * scale_dyn_range
        scale = mx / (2**(quantWgtStoreBits - 1) - 1)

        mult_mx = 2**(quantWgtStoreBits - 1) - 1
        mult_scale = mult_mx / (2**(quantBits - 1) - 1)

        self.register_buffer('sw', scale * mult_scale)
        self.weight.data = torch.round(self.weight / (scale + 1e-32))

  def forward(self, input):

    if quantFWDWgt == 'sawb':
      raise Exception('not implemented')
      w_q = UniformQuantizeSawb.apply(
          self.weight, self.c1, self.c2, self.QpW, self.QnW)
    elif quantFWDWgt == 'int':
      w_q, sw = dynamic_intQ_FWD.apply(self.weight)
    elif quantFWDWgt == 'lsq':
      raise Exception('not implemented')
      w_q = lsq(self.weight, self.lsq_wgt, self.QpW, self.QnW)
    elif quantFWDWgt == 'mem':
      # make sure grad updates
      # with torch.no_grad():
      #   tmp_w = torch.clip(
      #     self.weight, -(2**(quantWgtStoreBits - 1) - 1), (2**(quantWgtStoreBits - 1) - 1))
      #   tmp_w = torch.round(tmp_w)

      #   self.weight.data = tmp_w

      w_q = self.weight
      sw = self.sw
    elif quantFWDWgt == 'noq':
      w_q = self.weight
      sw = torch.tensor([1.0])
    else:
      raise Exception(
          'FWD weight quantized method not implemented: ' + quantFWDWgt)

    if torch.min(input) < 0:
      self.QnA = -2 ** (self.abits - 1) + 1
      self.QpA = 2 ** (self.wbits - 1) - 1

    if quantFWDAct == 'sawb':
      raise Exception('not implemented')
      qinput = UniformQuantizeSawb.apply(
          input, self.c1, self.c2, self.QpA, self.QnA)
    elif quantFWDAct == 'int':

      # qinput, sa = [], []
      # for i in range(int(np.ceil(input.shape[1]/quantBlockSize))):
      #   ta, tsa = dynamic_intQ_FWD.apply(input[:,i*quantBlockSize:(i+1)*quantBlockSize])
      #   qinput.append(ta)
      #   sa.append(tsa)

      # qinput = torch.cat(qinput, dim=1)
      # sa = torch.stack(sa)
      
      qinput, sa = dynamic_intQ_FWD.apply(input)
    elif quantFWDAct == 'lsq':
      raise Exception('not implemented')
      qinput = lsq(input, self.lsq_wgt, self.QpA, self.QnA)
    elif quantFWDAct == 'noq':
      qinput = input
      sa = torch.tensor([1.0])
    else:
      raise Exception(
          'FWD act quantized method not implemented: ' + quantFWDWgt)

    # TODO: optimize speed of hadamard creation
    if input.shape[0] != quantBatchSize:
      h_bs = torch.tensor(make_hadamard(
          input.shape[0]), dtype=self.weight.dtype).to(self.weight.device)
    else:
      h_bs = self.hadamard_bs

    global current_uname
    current_uname = self.uname

    output = FLinearQ.apply(qinput, w_q, self.hadamard_out, h_bs, sa, sw) 
    if self.bias is not None:
      output += self.bias

    return {'logits': output}


class Conv2d_Ours(nn.Conv2d):

  """docstring for Conv2d_BF16."""

  def __init__(self, uname, *args, **kwargs):
    super(Conv2d_Ours, self).__init__(*args, **kwargs)
    init_properties(self, uname)

    self.register_buffer('hadamard_out', torch.tensor(
        make_hadamard(self.out_channels), dtype=self.weight.dtype))
    self.register_buffer('hadamard_bs', torch.tensor(
        torch.zeros((1, 1)), dtype=self.weight.dtype))

    # self.lsq_act = Parameter(torch.tensor([1.], dtype=torch.float32))
    # self.lsq_wgt = Parameter(torch.tensor(
    #     [self.weight.abs().mean() * 2 / np.sqrt(self.QpW)], dtype=torch.float32))

    if quantFWDWgt == 'mem':
      # if 'backbone' not in uname:
      #   scale_dyn_range = global_args["init_dyn_scale"]
      # else:
      scale_dyn_range = global_args["dyn_scale"]
      with torch.no_grad():

        mx = self.weight.abs().max() * scale_dyn_range
        scale = mx / (2**(quantWgtStoreBits - 1) - 1)

        mult_mx = 2**(quantWgtStoreBits - 1) - 1
        mult_scale = mult_mx / (2**(quantBits - 1) - 1)

        self.register_buffer('sw', scale * mult_scale)
        self.weight.data = torch.round(self.weight / (scale + 1e-32))

  def forward(self, input):

    # if self.quantizeFwd:
    if quantFWDWgt == 'sawb':
      raise Exception('not implemented')
      w_q = UniformQuantizeSawb.apply(
          self.weight, self.c1, self.c2, self.QpW, self.QnW)
    elif quantFWDWgt == 'int':
      w_q, sw = dynamic_intQ_FWD.apply(self.weight)
    elif quantFWDWgt == 'lsq':
      raise Exception('not implemented')
      w_q = lsq(self.weight, self.lsq_wgt, self.QpW, self.QnW)
    elif quantFWDWgt == 'mem':
      # make sure grad updates
      # with torch.no_grad():
      #   # tmp_w = torch.clip(self.weight, -(2**(quantWgtStoreBits-1)-1), (2**(quantWgtStoreBits-1)-1))
      #   # tmp_w = torch.round(tmp_w)

      #   self.weight.data = tmp_w

      w_q = self.weight
      sw = self.sw
    elif quantFWDWgt == 'noq':
      w_q = self.weight
      sw = torch.tensor([1.0])
    else:
      raise Exception(
          'FWD weight quantized method not implemented: ' + quantFWDWgt)

    if torch.min(input) < 0:
      self.QnA = -2 ** (self.abits - 1) + 1
      self.QpA = 2 ** (self.wbits - 1) - 1

    if quantFWDAct == 'sawb':
      raise Exception('not implemented')
      qinput = UniformQuantizeSawb.apply(
          input, self.c1, self.c2, self.QpA, self.QnA)
    elif quantFWDAct == 'int':
      qinput, sa = dynamic_intQ_FWD.apply(input)
    elif quantFWDAct == 'lsq':
      raise Exception('not implemented')
      qinput = lsq(input, self.lsq_wgt, self.QpA, self.QnA)
    elif quantFWDAct == 'noq':
      qinput = input
      sa = torch.tensor([1.0])
    else:
      raise Exception(
          'FWD act quantized method not implemented: ' + quantFWDWgt)

    # TODO: optimize speed of hadamard creation

    qinput = torch.nn.functional.unfold(
        qinput, self.kernel_size, padding=self.padding, stride=self.stride).transpose(1, 2)
    w_q = w_q.view(w_q.size(0), -1).t()

    if self.hadamard_bs.sum() == 0:
      self.hadamard_bs = torch.tensor(make_hadamard(
          qinput.shape[1]), dtype=self.weight.dtype).to(self.weight.device)

    # flinearq_fn = torch.vmap(FLinearQ.apply, randomness='different')
    # out = flinearq_fn(qinput, w_q.T.unsqueeze(0).repeat(qinput.shape[0], 1, 1), self.hadamard_out.unsqueeze(
    #     0).repeat(qinput.shape[0], 1, 1), self.hadamard_bs.unsqueeze(0).repeat(qinput.shape[0], 1, 1),
    #     sa.unsqueeze(0).repeat(qinput.shape[0], 1, 1), sw.unsqueeze(0).repeat(qinput.shape[0], 1, 1))

    out = []
    for i in range(qinput.shape[0]):
      out.append(FLinearQ.apply(qinput[i,:], w_q.T, self.hadamard_out, self.hadamard_bs, sa, sw))
    out = torch.stack(out)

    # reshaping outputs into image form with batch, channel, height, width
    out = out.transpose(1, 2)
    output = out.view((input.shape[0], self.out_channels, int(
        input.shape[-2] / self.stride[0]), int(input.shape[-1] / self.stride[1])))

    if self.bias is not None:
      output += self.bias.view(1, -1, 1, 1)

    # np.testing.assert_allclose(output_cmp.cpu().detach().numpy(), output.cpu().detach().numpy(), rtol=1e-05, atol=1e-2)

    # else:
    #   output = F.linear(input, self.weight, self.bias)

    return output
