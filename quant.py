# IMSL Lab - University of Notre Dame | University of St Andrews
# Author: Clemens JS Schaefer | Martin Schiemer
# Quantized training.

import scipy
import numpy as np
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.autograd.function import Function
from torch.autograd.function import InplaceFunction

from backbones.linears import SimpleLinear

from squant_function import SQuant_func

from hadamard import make_hadamard, biggest_power2_factor


track_stats = {'grads': {}, 'acts': {}, 'wgts': {},
               'grad_stats': {}, 'test_acc': [], 'train_acc': [], 'loss': []}
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
global_args = None

QpW = None
QnW = None
QpA = None
QnA = None

quantGradMxScale = 1.

scale_library = {'a': {}, 'w': {}, 'g': {}}


class QuantMomentumOptimizer(torch.optim.Optimizer):
      
  # Init Method:
  def __init__(self, params, lr=1e-3, momentum=0.9):
    super(QuantMomentumOptimizer, self).__init__(params, defaults={'lr': lr})
    self.momentum = momentum
    self.state = dict()
    for group in self.param_groups:
      for p in group['params']:
        if p.grad is not None:
          self.state[p] = dict(mom=torch.zeros_like(p.data))
    
  # Step Method
  def step(self):
    for group in self.param_groups:
      for p in group['params']:
        if p not in self.state:
          self.state[p] = dict(mom=torch.zeros_like(p.data))
        if p.grad is not None:

          p.data -= group['lr'] * p.grad.data

          p.data = torch.clip(p.data, -(2**(quantWgtStoreBits-1)-1), (2**(quantWgtStoreBits-1)-1))
          p.data = torch.round(p.data)


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


def place_track(m, layer_list, c_path, lin_w, lin_b):
  track_stats['test_acc'] = []
  track_stats['train_acc'] = []
  track_stats['loss'] = []
  for attr_str in dir(m):
    target_attr = getattr(m, attr_str)
    if isinstance(target_attr, nn.Conv2d):
      if not hasattr(target_attr, 'c1'):
        if c_path + '_' + attr_str in layer_list:
          track_stats['grad_stats'][c_path + '_'
                                    + attr_str] = {'max': [], 'min': [], 'norm': [], 'mean': []}
          track_stats['grads'][c_path + '_' + attr_str] = []
          track_stats['acts'][c_path + '_' + attr_str] = []
          track_stats['wgts'][c_path + '_' + attr_str] = []
          setattr(m, attr_str,
                  Conv2d_track(track_name=c_path + '_' + attr_str,
                               in_channels=target_attr.in_channels,
                               out_channels=target_attr.out_channels,
                               kernel_size=target_attr.kernel_size,
                               stride=target_attr.stride,
                               padding=target_attr.padding,
                               padding_mode=target_attr.padding_mode,
                               dilation=target_attr.dilation,
                               groups=target_attr.groups,
                               bias=hasattr(target_attr, 'bias'),))
    if isinstance(target_attr, nn.Linear) or isinstance(target_attr,
                                                        SimpleLinear):
      if c_path + '_' + attr_str in layer_list:
        track_stats['grad_stats'][c_path + '_'
                                  + attr_str] = {'max': [], 'min': [], 'norm': [], 'mean': []}
        track_stats['grads'][c_path + '_' + attr_str] = []
        track_stats['acts'][c_path + '_' + attr_str] = []
        track_stats['wgts'][c_path + '_' + attr_str] = []
        setattr(m, attr_str,
                Linear_track(track_name=c_path + '_' + attr_str,
                             in_features=target_attr.in_features,
                             out_features=target_attr.out_features,
                             bias=hasattr(target_attr, 'bias'),))
        if lin_w is not None:
          m.fc.weight = nn.Parameter(lin_w)
        if lin_b is not None:
          m.fc.bias = nn.Parameter(lin_b)
  for n, ch in m.named_children():
    place_track(ch, layer_list, c_path + '_' + n, lin_w, lin_b)


def place_quant(m, lin_w, lin_b, c_path='',):

  global global_args
  global_args = m.args

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
                                        bias=getattr(target_attr, 'bias') is not None,
                                        uname=c_path + '_' + attr_str,))
          
          if lin_w is not None and attr_str == 'fc':

            if quantFWDWgt == 'mem':
              if old_sw is None:
                pass
              else:
                scale_dyn_range = global_args["dyn_scale"]
                lin_w = lin_w/scale_dyn_range
                lin_w = torch.round(lin_w)
                m.fc.sw.data = old_sw*scale_dyn_range
                m.fc.weight.data = lin_w
            else:
              m.fc.weight.data = lin_w
          if lin_b is not None and attr_str == 'fc':
            m.fc.bias = nn.Parameter(lin_b)
  for n, ch in m.named_children():
    place_quant(ch, lin_w, lin_b, c_path + '_' + n,)


def save_lin_params(m):
  weights = None
  bias = None
  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      if isinstance(target_attr, SimpleLinear) or isinstance(target_attr,
                                                             Linear_LUQ):
        weights = target_attr.weight
        bias = target_attr.bias
  for n, ch in m.named_children():
    save_lin_params(ch)

  return weights, bias

# quant code based on supplementary materials of
# https://openreview.net/forum?id=yTbNYYcopd


class Conv2d_track(nn.Conv2d):

  """docstring for Conv2d_BF16."""

  def __init__(self, track_name, *args, **kwargs):
    super(Conv2d_track, self).__init__(*args, **kwargs)
    self.track_name = track_name

  def forward(self, input):

    track_stats['wgts'][self.track_name] = (
        self.weight.mean(), self.weight.max(), self.weight.min())
    track_stats['acts'][self.track_name] = (
        input.mean(), input.max(), input.min())

    output = F.conv2d(input, self.weight, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

    output = GradTrack.apply(
        output, self.track_name)
    return output


class Linear_track(nn.Linear):

  """docstring for Conv2d_BF16."""

  def __init__(self, track_name, *args, **kwargs):
    super(Linear_track, self).__init__(*args, **kwargs)
    self.track_name = track_name

  def forward(self, input):

    track_stats['wgts'][self.track_name] = (
        self.weight.mean(), self.weight.max(), self.weight.min())
    track_stats['acts'][self.track_name] = (
        input.mean(), input.max(), input.min())

    output = F.linear(input, self.weight, self.bias)

    output = GradTrack.apply(
        output, self.track_name)
    return {'logits': output}


class GradTrack(Function):

  @staticmethod
  def forward(ctx, x, name):
    ctx.name = name

    return x

  @staticmethod
  def backward(ctx, grad_output):

    size_total = len(grad_output.flatten())
    track_stats['grads'][ctx.name].append(grad_output.flatten(
    )[torch.randint(size_total, (int(size_total * .01),))].cpu())

    # grad norm
    track_stats['grad_stats'][ctx.name]['norm'].append(
        torch.linalg.vector_norm(grad_output.flatten()).cpu())

    # grad max
    track_stats['grad_stats'][ctx.name]['max'].append(grad_output.max().cpu())

    # grad min
    track_stats['grad_stats'][ctx.name]['min'].append(grad_output.min().cpu())

    # grad mean
    track_stats['grad_stats'][ctx.name]['mean'].append(
        grad_output.mean().cpu())

    return grad_output, None

# begin code from:
# from https://openreview.net/pdf?id=yTbNYYcopd suppl. materials


class UniformQuantizeSawb(InplaceFunction):

  @staticmethod
  def forward(ctx, input, c1, c2, Qp, Qn):

    output = input.clone()

    with torch.no_grad():
      clip = (c1 * torch.sqrt(torch.mean(input**2))) - \
          (c2 * torch.mean(input.abs()))
      scale = 2 * clip / (Qp - Qn)

      output.div_(scale)
      output.clamp_(Qn, Qp).round_()
      output.mul_(scale)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    # straight-through estimator
    grad_input = grad_output
    return grad_input, None, None, None, None


class Linear_LUQ(nn.Linear):

  """docstring for Conv2d_BF16."""

  def __init__(self, uname, *args, **kwargs):
    super(Linear_LUQ, self).__init__(*args, **kwargs)
    init_properties(self, uname)

  def forward(self, input):

    if False: # self.quantizeFwd:

      w_q = UniformQuantizeSawb.apply(
          self.weight, self.c1, self.c2, self.QpW, self.QnW)

      if torch.min(input) < 0:
        self.QnA = -2 ** (self.abits - 1) + 1
        self.QpA = 2 ** (self.wbits - 1) - 1

      qinput = UniformQuantizeSawb.apply(
          input, self.c1, self.c2, self.QpA, self.QnA)

      # all
      output = F.linear(qinput, w_q, self.bias)

    else:
      output = F.linear(input, self.weight, self.bias)

    output = GradStochasticClippingQ.apply(
        output, self.quantizeBwd, self.layerIdx, self.repeatBwd, self.uname)
    return {'logits': output}


class Conv2d_LUQ(nn.Conv2d):

  """docstring for Conv2d_BF16."""

  def __init__(self, uname, *args, **kwargs):
    super(Conv2d_LUQ, self).__init__(*args, **kwargs)
    init_properties(self, uname)

  def forward(self, input):

    if False: # self.quantizeFwd:
      w_q = UniformQuantizeSawb.apply(
          self.weight, self.c1, self.c2, self.QpW, self.QnW)

      if torch.min(input) < 0:
        self.QnA = -2 ** (self.abits - 1) + 1
        self.QpA = 2 ** (self.wbits - 1) - 1

      qinput = UniformQuantizeSawb.apply(
          input, self.c1, self.c2, self.QpA, self.QnA)

      # all
      output = F.conv2d(qinput, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    else:
      output = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    output = GradStochasticClippingQ.apply(
        output, self.quantizeBwd, self.layerIdx, self.repeatBwd, self.uname)
    return output


class GradStochasticClippingQ(Function):

  @staticmethod
  def forward(ctx, x, quantizeBwd, layerIdx, repeatBwd, uname):
    ctx.uname = uname
    ctx.save_for_backward(torch.tensor(quantizeBwd),
                          torch.tensor(layerIdx), torch.tensor(repeatBwd))
    return x

  @staticmethod
  def backward(ctx, grad_output):
    quant, layerIdx, repeatBwd = ctx.saved_tensors
    if quant:
      out = []
      for i in range(repeatBwd):

        mx = torch.max(grad_output.abs())

        bits = quantBits - 1
        # was 1 before, need to be 2 centered around 0
        alpha = mx / 2**(2**bits - 2)

        if quantBWDGrad1 == "stochastic":
          alphaEps = alpha * \
              torch.rand(grad_output.shape, device=grad_output.device)
        else:
          alphaEps = alpha

        grad_abs = grad_output.abs()

        grad_input = torch.where(
            grad_abs < alpha, alpha * torch.sign(grad_output), grad_output)

        grad_input = torch.where(grad_abs < alphaEps, torch.tensor(
            [0], dtype=torch.float32, device=grad_output.device), grad_input)

        grad_inputQ = grad_input.clone()

        if quantBWDGrad1 == "stochastic":
          noise = (2 ** torch.floor(torch.log2((grad_inputQ.abs() / alpha)))
                   ) * grad_inputQ.new(grad_inputQ.shape).uniform_(-0.5, 0.5)
        else:
          noise = torch.zeros_like(grad_inputQ)

        grad_inputQ = 2 ** torch.floor(torch.log2(
            ((grad_inputQ.abs() / alpha) + noise) * 4 / 3)) * alpha

        grad_inputQ = torch.sign(grad_input) * torch.where(grad_inputQ < (alpha * (2 ** torch.floor(torch.log2(((grad_input.abs() / alpha)))))), 
                                                           alpha * (2 ** torch.floor(torch.log2(((grad_input.abs() / alpha))))), grad_inputQ)
        grad_inputQ = torch.where(grad_input == 0, torch.tensor(
            [0], dtype=torch.float, device=grad_output.device), grad_inputQ)

        out.append(grad_inputQ)
        # assert grad_inputQ.unique().shape[0] <= 2**quantBits
      grad_input = sum(out) / repeatBwd

    else:
      grad_input = grad_output

    return grad_input, None, None, None, None


def dynamic_intQ(x):
  # can only be used in BWD - not differentiable
  mx = x.abs().max() * global_args["quantile"] # torch.quantile(x.abs(), .99) # TODO optimize calibration
  scale = mx/(2**(quantBits-1)-1)
  x = torch.clamp(x, -mx, mx)
  return torch.round(x/(scale + 1e-32)), scale # epsilion for vmap # TODO eps size?


def dynamic_squant(x, scale = 1):
  dim = x.shape
  x = x.flatten()

  mx = x.abs().max() * scale # torch.quantile(x.abs(), .99) # TODO optimize calibration
  scale = mx/(2**(quantBits-1)-1)

  x_clamp = torch.clamp(x, -mx, mx)/(scale + 1e-32)
  xq = torch.round(x_clamp)

  xq_down = xq - 1
  xq_up = xq + 1

  qe = xq - x_clamp

  # bias meaning positive too much round up
  bias = qe.sum().floor()
  bias_use = torch.where(bias.abs() -1 <= 0, torch.tensor([1], device=x.device), bias)

  qe_cutoff_pos = torch.gather(torch.sort(qe, descending = True).values, 0, bias_use.abs().type(torch.int64) -1)
  xq_pos = torch.where(qe < qe_cutoff_pos, xq, xq_down)

  qe_cutoff_neg = torch.gather(torch.sort(-qe, descending = True).values, 0, bias_use.abs().type(torch.int64) -1)
  xq_neg = torch.where(-qe < qe_cutoff_neg, xq, xq_up)

  xqt = torch.where(bias > 0, xq_pos, xq_neg)
  xq = torch.where(bias.abs()-1 <= 0, xq, xqt)

  # assert torch.unique(xq).shape[0] < 9

  xq = torch.reshape(xq, dim)
  return xq * scale

def dynamic_stoch(x, scale = 1):
  # can only be used in BWD - not differentiable
  dim = x.shape
  x = x.flatten()

  mx = x.abs().max() * scale # torch.quantile(x.abs(), .99) # TODO optimize calibration
  scale = mx/(2**(quantBits-1)-1)
  x = torch.clamp(x, -mx, mx)/(scale + 1e-32)

  sign = torch.sign(x)
  xq_ord = torch.floor(x.abs())

  qe = x.abs() - xq_ord

  flip = torch.bernoulli(qe)

  xq = xq_ord + flip

  xq = torch.reshape(sign * xq, dim)
  return xq, scale

class dynamic_intQ_FWD(Function):

  @staticmethod
  def forward(ctx, x):
    mx = x.abs().max() * global_args["quantile"] # torch.quantile(x.abs(), .99) # TODO optimize calibration
    ctx.mx = mx
    ctx.save_for_backward(x)
    if x.min() < 0:
      scale = mx/(2**(quantBits-1)-1)
      x = torch.clamp(x, -mx, mx)
    else:
      scale =  mx/(2**(quantBits)-1)
      x = torch.clamp(x, 0, mx)
    
    return torch.round(x/scale), scale

  @staticmethod
  def backward(ctx, grad_output, grad_scale):
    # STE
    x, = ctx.saved_tensors

    # TODO local scale reconsideration
    # local_mx = ctx.mx
    local_mx = (2**(quantBits-1)-1)

    grad_output = torch.where(x>local_mx, torch.tensor([0], dtype=x.dtype, device=x.device), grad_output)
    grad_output = torch.where(x<-local_mx, torch.tensor([0], dtype=x.dtype, device=x.device), grad_output)
    return grad_output, None

def SAWB(input, c1, c2, Qp, Qn):
  output = input.clone()

  clip = (c1 * torch.sqrt(torch.mean(input**2))) - \
      (c2 * torch.mean(input.abs()))
  scale = 2 * clip / (Qp - Qn)

  output.div_(scale)
  output.clamp_(Qn, Qp).round_()
  output.mul_(scale)

  return output

class FLinearQ(torch.autograd.Function):
  generate_vmap_rule = True

  @staticmethod
  def forward(x, w, h_out, h_bs, sx, sw):

    if quantFWDWgt == 'mem':
      # only get quantFWDWgt
      mx = 2**(quantWgtStoreBits-1)-1
      scale = mx/(2**(quantBits-1)-1)
      w = torch.clamp(w, -mx, mx)
      w = torch.round(w/(scale + 1e-32))

    output = F.linear(x, w)

    # requantize to int16 (clamp to big values - no scale)
    n = 2**quantAccBits/2 - 1
    output = torch.clamp(output, -n, n)

    return output * sw * sx

  @staticmethod
  def setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])

  @staticmethod
  def backward(ctx, grad_output):

    # TODO scales
    x, w, h_out, h_bs, sx, sw = ctx.saved_tensors

    w_h1 = h_out @ w
    # requantize weights
    if quantBWDWgt == 'int':
      w_h1, swh1 = dynamic_intQ(w_h1)
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
    n = 2**quantAccBits/2 - 1
    grad_input = torch.clamp(grad_input, -n, n) * 1 / biggest_power2_factor(h_out.shape[0])

    x_h2 = h_bs @ x
    # requantize acts
    if quantBWDAct == 'int':
      x_h2, sxh2 = dynamic_intQ(x_h2)
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
    n = 2**quantAccBits/2 - 1
    grad_w = torch.clamp(grad_w, -n, n)

    # 
    return grad_input * sg1 * swh1 * sw, grad_w * sg2 * sxh2 * sx * 1 / biggest_power2_factor(h_bs.shape[0]), None, None, None, None


def grad_scale(x, scale):
  # https://github.com/zhutmost/lsq-net
  y = x
  y_grad = x * scale
  return (y - y_grad).detach() + y_grad


def round_pass(x):
  # https://github.com/zhutmost/lsq-net
  y = x.round()
  y_grad = x
  return (y - y_grad).detach() + y_grad


def lsq(x, s, Qp, Qn):
  # https://github.com/zhutmost/lsq-net
  s_grad_scale = 1.0 / ((Qp * x.numel()) ** 0.5)
  s_scale = grad_scale(s, s_grad_scale)

  x = x / s_scale
  x = torch.clamp(x, Qn, Qp)
  x = round_pass(x)
  x = x * s_scale

  return x


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
      if 'backbone' not in uname:
        scale_dyn_range = global_args["init_dyn_scale"]
      else:
        scale_dyn_range = global_args["dyn_scale"]

      # DEBUG!!!!
      # scale_dyn_range = .975
      # DEBUG!!!!

      with torch.no_grad():

        mx = self.weight.abs().max() * scale_dyn_range
        scale = mx/(2**(quantWgtStoreBits-1)-1)

        mult_mx = 2**(quantWgtStoreBits-1)-1
        mult_scale = mult_mx/(2**(quantBits-1)-1)

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
      with torch.no_grad():
        tmp_w = torch.clip(self.weight, -(2**(quantWgtStoreBits-1)-1), (2**(quantWgtStoreBits-1)-1))
        tmp_w = torch.round(tmp_w)

        self.weight.data = tmp_w

      w_q = self.weight
      sw = self.sw
    elif quantFWDWgt == 'noq':
      w_q = self.weight
      sw = torch.tensor([1.0])
    else:
      raise Exception('FWD weight quantized method not implemented: ' + quantFWDWgt)


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
      raise Exception('FWD act quantized method not implemented: ' + quantFWDWgt)

    # TODO: optimize speed of hadamard creation
    if input.shape[0] != quantBatchSize:
      h_bs = torch.tensor(make_hadamard(
          input.shape[0]), dtype=self.weight.dtype).to(self.weight.device)
    else:
      h_bs = self.hadamard_bs

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
      if 'backbone' not in uname:
        scale_dyn_range = global_args["init_dyn_scale"]
      else:
        scale_dyn_range = global_args["dyn_scale"]
      with torch.no_grad():

        mx = self.weight.abs().max() * scale_dyn_range
        scale = mx/(2**(quantWgtStoreBits-1)-1)

        self.register_buffer('sw', scale)
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
      with torch.no_grad():
        # tmp_w = torch.clip(self.weight, -(2**(quantWgtStoreBits-1)-1), (2**(quantWgtStoreBits-1)-1))
        # tmp_w = torch.round(tmp_w)

        self.weight.data = tmp_w

      w_q = self.weight
      sw = self.sw
    elif quantFWDWgt == 'noq':
      w_q = self.weight
      sw = torch.tensor([1.0])
    else:
      raise Exception('FWD weight quantized method not implemented: ' + quantFWDWgt)

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
      raise Exception('FWD act quantized method not implemented: ' + quantFWDWgt)

    # TODO: optimize speed of hadamard creation

    qinput = torch.nn.functional.unfold(
        qinput, self.kernel_size, padding=self.padding, stride=self.stride).transpose(1, 2)
    w_q = w_q.view(w_q.size(0), -1).t()

    if self.hadamard_bs.sum() == 0:
      self.hadamard_bs = torch.tensor(make_hadamard(
          qinput.shape[1]), dtype=self.weight.dtype).to(self.weight.device)

    flinearq_fn = torch.vmap(FLinearQ.apply, randomness = 'different')
    out = flinearq_fn(qinput, w_q.T.unsqueeze(0).repeat(qinput.shape[0], 1, 1), self.hadamard_out.unsqueeze(
        0).repeat(qinput.shape[0], 1, 1), self.hadamard_bs.unsqueeze(0).repeat(qinput.shape[0], 1, 1),
    sa.unsqueeze(0).repeat(qinput.shape[0], 1, 1), sw.unsqueeze(0).repeat(qinput.shape[0], 1, 1))

    # out = []
    # for i in range(qinput.shape[0]):
    #   out.append(FLinearQ.apply(qinput[i,:], w_q.T, self.hadamard_out, self.hadamard_bs))
    # out = torch.stack(out)

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
