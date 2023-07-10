# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Quantized training.

import math
import dataclasses
from collections.abc import Callable
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd.function import Function
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction

from convs.linears import SimpleLinear

track_stats = {}
calibrate_phase = False

scale_library = {'a':{}, 'w':{}, 'g':{}}

def place_track(m, layer_list, c_path, lin_w, lin_b):
  for attr_str in dir(m):
    target_attr = getattr(m, attr_str)
    if isinstance(target_attr, nn.Conv2d):
      if not hasattr(target_attr, 'c1'):
        print(c_path+'_'+attr_str)
        if c_path+'_'+attr_str in layer_list:
          track_stats[c_path+'_'+attr_str] = []
          setattr(m, attr_str, Conv2d_track(track_name = c_path+'_'+attr_str,
                                            in_channels=target_attr.in_channels,
                                            out_channels=target_attr.out_channels,
                                            kernel_size=target_attr.kernel_size,
                                            stride=target_attr.stride,
                                            padding=target_attr.padding,
                                            padding_mode=target_attr.padding_mode,
                                            dilation=target_attr.dilation,
                                            groups=target_attr.groups,
                                            bias=hasattr(target_attr, 'bias'),))
    if isinstance(target_attr, nn.Linear) or isinstance(target_attr, SimpleLinear):
      print(c_path+'_'+attr_str)
      if c_path+'_'+attr_str in layer_list:
        track_stats[c_path+'_'+attr_str] = []
        setattr(m, attr_str, Linear_track(track_name = c_path+'_'+attr_str,
                                          in_features = target_attr.in_features,
                                          out_features = target_attr.out_features,
                                          bias=hasattr(target_attr, 'bias'),))
        if lin_w is not None:
          m.fc.weight = nn.Parameter(lin_w)
        if lin_b is not None:
          m.fc.bias = nn.Parameter(lin_b)
  for n, ch in m.named_children():
    place_track(ch, layer_list, c_path + '_' + n, lin_w, lin_b)


def place_quant(m, lin_w, lin_b,c_path = '',):
  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      if isinstance(target_attr, nn.Conv2d):
        if not hasattr(target_attr, 'c1'):
          setattr(m, attr_str, Conv2d_LUQ(in_channels=target_attr.in_channels,
                                          out_channels=target_attr.out_channels,
                                          kernel_size=target_attr.kernel_size,
                                          stride=target_attr.stride,
                                          padding=target_attr.padding,
                                          padding_mode=target_attr.padding_mode,
                                          dilation=target_attr.dilation,
                                          groups=target_attr.groups,
                                          bias=hasattr(target_attr, 'bias'),
                                          uname = c_path+'_'+attr_str,))
      if isinstance(target_attr, nn.Linear) or isinstance(target_attr, SimpleLinear):
        setattr(m, attr_str, Linear_LUQ(in_features = target_attr.in_features,
                                        out_features = target_attr.out_features,
                                        bias=hasattr(target_attr, 'bias'),
                                        uname = c_path+'_'+attr_str,))
        if lin_w is not None:
          m.fc.weight = nn.Parameter(lin_w)
        if lin_b is not None:
          m.fc.bias = nn.Parameter(lin_b)
  for n, ch in m.named_children():
    place_quant(ch, lin_w, lin_b,  c_path + '_' + n,)


def save_lin_params(m):
  weights = None
  bias = None
  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      if isinstance(target_attr, SimpleLinear) or isinstance(target_attr, Linear_LUQ):
        weights = target_attr.weight
        bias = target_attr.bias
  for n, ch in m.named_children():
    save_lin_params(ch)

  return weights, bias

# quant code losely based on
# https://github.com/EunhyeokPark/PROFIT/blob/master/quant_op/duq.py


class Conv2d_track(nn.Conv2d):

  """docstring for Conv2d_BF16."""

  def __init__(self, track_name, *args, **kwargs):
    super(Conv2d_track, self).__init__(*args, **kwargs)
    self.track_name = track_name

  def forward(self, input):


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

    
    output = F.linear(input, self.weight, self.bias)

    output = GradTrack.apply(
        output, self.track_name)
    return  {'logits': output}


class GradTrack(Function):

  @staticmethod
  def forward(ctx, x, name):
    ctx.name = name

    return x

  @staticmethod
  def backward(ctx, grad_output):
    
    size_total = len(grad_output.flatten())
    track_stats[ctx.name].append(grad_output.flatten()[torch.randint(size_total, ( int(size_total * .01) ,))].cpu())

    return grad_output, None

# begin code from:
# from https://openreview.net/pdf?id=yTbNYYcopd suppl. materials

# class UniformQuantizeSawb(InplaceFunction):

#   @staticmethod
#   def forward(ctx, input, c1, c2, Qp, Qn):

#     output = input.clone()

#     with torch.no_grad():
#       clip = (c1 * torch.sqrt(torch.mean(input**2))) - \
#           (c2 * torch.mean(input.abs()))
#       scale = 2 * clip / (Qp - Qn)

#       # import pdb; pdb.set_trace()
#       print(scale)
#       print(clip)
#       output.div_(scale)
#       output.clamp_(Qn, Qp).round_()
#       output.mul_(scale)
#     return output

#   @staticmethod
#   def backward(ctx, grad_output):
#     # straight-through estimator
#     grad_input = grad_output
#     return grad_input, None, None, None, None


class UniformQuantizeSawb(InplaceFunction):

  @staticmethod
  def forward(ctx, input, Qp, Qn, scale, clip):

    output = input.clone()

    with torch.no_grad():
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
    self.fullName = ''
    self.statistics = []
    self.layerIdx = 0

    self.alpha = Parameter(torch.tensor([1], dtype=torch.float32))
    self.beta = Parameter(torch.tensor([1], dtype=torch.float32))
    self.abits = 4
    self.wbits = 4

    self.QnW = -2 ** (self.wbits - 1)
    self.QpW = 2 ** (self.wbits - 1)
    self.QnA = 0
    self.QpA = 2 ** self.abits - 1

    self.register_buffer('init_stateW', torch.zeros(1))
    self.register_buffer('init_stateA', torch.zeros(1))

    self.register_buffer('gradScaleW', torch.zeros(1))
    self.register_buffer('gradScaleA', torch.zeros(1))

    self.quantizeFwd = True
    self.quantizeBwd = True

    self.c1 = 12.1
    self.c2 = 12.2
    self.stochastic = True
    self.repeatBwd = 1

    self.uname = uname


  def forward(self, input):

    if self.quantizeFwd:
      if calibrate_phase:
        with torch.no_grad():
          if self.uname not in scale_library['w']: 
            scale_library['w'][self.uname] = {}
            scale_library['w'][self.uname]['clip_w'] = (self.c1 * torch.sqrt(torch.mean(self.weight**2))) - (self.c2 * torch.mean(self.weight.abs()))
            scale_library['w'][self.uname]['scale_w']= 2 * scale_library['w'][self.uname]['clip_w'] / (self.QpW - self.QnW)
          else:
            scale_library['w'][self.uname]['clip_w'] = scale_library['w'][self.uname]['clip_w'] * .9 + .1 * ( (self.c1 * torch.sqrt(torch.mean(self.weight**2))) - (self.c2 * torch.mean(self.weight.abs())))
            scale_library['w'][self.uname]['scale_w'] = scale_library['w'][self.uname]['scale_w'] * .9 + .1 * ( 2 * scale_library['w'][self.uname]['clip_w'] / (self.QpW - self.QnW))
        w_q = self.weight
      else:

        w_q = UniformQuantizeSawb.apply(self.weight, self.QpW, self.QnW, scale_library['w'][self.uname]['scale_w'], scale_library['w'][self.uname]['clip_w'])
      # w_q = UniformQuantizeSawb.apply(self.weight, self.c1, self.c2, self.QpW, self.QnW)

      if torch.min(input) < 0:
        self.QnA = -2 ** (self.abits - 1)


      if calibrate_phase:
        with torch.no_grad():
          if self.uname not in scale_library['a']:
            scale_library['a'][self.uname] = {}
            scale_library['a'][self.uname]['clip_a'] = (self.c1 * torch.sqrt(torch.mean(input**2))) - (self.c2 * torch.mean(input.abs()))
            scale_library['a'][self.uname]['scale_a']= 2 * scale_library['a'][self.uname]['clip_a'] / (self.QpA - self.QnA)
          else:
            scale_library['a'][self.uname]['clip_a'] = scale_library['a'][self.uname]['clip_a'] * .9 + .1 * ( (self.c1 * torch.sqrt(torch.mean(input**2))) - (self.c2 * torch.mean(input.abs())))
            scale_library['a'][self.uname]['scale_a'] = scale_library['a'][self.uname]['scale_a'] * .9 + .1 *  (2 * scale_library['a'][self.uname]['clip_a'] / (self.QpA - self.QnA))
        qinput = input
      else:
        qinput = UniformQuantizeSawb.apply(input, self.QpA, self.QnA, scale_library['a'][self.uname]['scale_a'], scale_library['a'][self.uname]['clip_a'])

      # qinput = UniformQuantizeSawb.apply(input, self.c1, self.c2, self.QpA, self.QnA)

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
    self.fullName = ''
    self.statistics = []
    self.layerIdx = 0

    self.alpha = Parameter(torch.tensor([1], dtype=torch.float32))
    self.beta = Parameter(torch.tensor([1], dtype=torch.float32))
    self.abits = 4
    self.wbits = 4

    self.QnW = -2 ** (self.wbits - 1)
    self.QpW = 2 ** (self.wbits - 1)
    self.QnA = 0
    self.QpA = 2 ** self.abits - 1

    self.register_buffer('init_stateW', torch.zeros(1))
    self.register_buffer('init_stateA', torch.zeros(1))

    self.register_buffer('gradScaleW', torch.zeros(1))
    self.register_buffer('gradScaleA', torch.zeros(1))

    self.quantizeFwd = True
    self.quantizeBwd = True

    self.c1 = 12.1
    self.c2 = 12.2
    self.stochastic = True
    self.repeatBwd = 1

    self.uname = uname

  def forward(self, input):


    if self.quantizeFwd:
      if calibrate_phase:
        with torch.no_grad():
          if self.uname not in scale_library['w']: 
            scale_library['w'][self.uname] = {}
            scale_library['w'][self.uname]['clip_w'] = (self.c1 * torch.sqrt(torch.mean(self.weight**2))) - (self.c2 * torch.mean(self.weight.abs()))
            scale_library['w'][self.uname]['scale_w']= 2 * scale_library['w'][self.uname]['clip_w'] / (self.QpW - self.QnW)
          else:
            scale_library['w'][self.uname]['clip_w'] = scale_library['w'][self.uname]['clip_w'] * .9 + .1 * ( (self.c1 * torch.sqrt(torch.mean(self.weight**2))) - (self.c2 * torch.mean(self.weight.abs())))
            scale_library['w'][self.uname]['scale_w'] = scale_library['w'][self.uname]['scale_w'] * .9 + .1 * ( 2 * scale_library['w'][self.uname]['clip_w'] / (self.QpW - self.QnW))
        w_q = self.weight
      else:

        w_q = UniformQuantizeSawb.apply(self.weight, self.QpW, self.QnW, scale_library['w'][self.uname]['scale_w'], scale_library['w'][self.uname]['clip_w'])

      if torch.min(input) < 0:
        self.QnA = -2 ** (self.abits - 1)

      if calibrate_phase:
        with torch.no_grad():
          if self.uname not in scale_library['a']:
            scale_library['a'][self.uname] = {}
            scale_library['a'][self.uname]['clip_a'] = (self.c1 * torch.sqrt(torch.mean(input**2))) - (self.c2 * torch.mean(input.abs()))
            scale_library['a'][self.uname]['scale_a']= 2 * scale_library['a'][self.uname]['clip_a'] / (self.QpA - self.QnA)
          else:
            scale_library['a'][self.uname]['clip_a'] = scale_library['a'][self.uname]['clip_a'] * .9 + .1 * ( (self.c1 * torch.sqrt(torch.mean(input**2))) - (self.c2 * torch.mean(input.abs())))
            scale_library['a'][self.uname]['scale_a'] = scale_library['a'][self.uname]['scale_a'] * .9 + .1 *  (2 * scale_library['a'][self.uname]['clip_a'] / (self.QpA - self.QnA))
        qinput = input
      else:
        qinput = UniformQuantizeSawb.apply(input, self.QpA, self.QnA, scale_library['a'][self.uname]['scale_a'], scale_library['a'][self.uname]['clip_a'])

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

        if calibrate_phase:
          mx = torch.max(grad_output)
          if ctx.uname not in scale_library['g']:
            scale_library['g'][ctx.uname] = mx
          else:
            scale_library['g'][ctx.uname] = .9 * scale_library['g'][ctx.uname] + .1 * mx
          return grad_output, None, None, None, None
        else:
          mx = scale_library['g'][ctx.uname]
          grad_output = torch.clamp(grad_output, max = mx)


        bits = 4
        alpha = mx / 2**(2**bits - 1)

        alphaEps = alpha * \
            torch.rand(grad_output.shape, device=grad_output.device)

        grad_abs = grad_output.abs()

        grad_input = torch.where(
            grad_abs < alpha, alpha * torch.sign(grad_output), grad_output)


        grad_input = torch.where(grad_abs < alphaEps, torch.tensor(
            [0], dtype=torch.float32, device=grad_output.device), grad_input)

        grad_inputQ = grad_input.clone()
        noise = (2 ** torch.floor(torch.log2((grad_inputQ.abs() / alpha)))
                 ) * grad_inputQ.new(grad_inputQ.shape).uniform_(-0.5, 0.5)
        grad_inputQ = 2 ** torch.floor(torch.log2(
            ((grad_inputQ.abs() / alpha) + noise) * 4 / 3)) * alpha

        grad_inputQ = torch.sign(grad_input) * torch.where(grad_inputQ < (alpha * (2 ** torch.floor(torch.log2(
            ((grad_input.abs() / alpha)))))), alpha * (2 ** torch.floor(torch.log2(((grad_input.abs() / alpha))))), grad_inputQ)
        grad_inputQ = torch.where(grad_input == 0, torch.tensor(
            [0], dtype=torch.float, device=grad_output.device), grad_inputQ)

        out.append(grad_inputQ)
      grad_input = sum(out) / repeatBwd

    else:
      grad_input = grad_output
    return grad_input, None, None, None, None


# end code from:
# from https://openreview.net/pdf?id=yTbNYYcopd suppl. materials


class RoundQuant(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, n_lv):
    return input.mul(n_lv - 1).round_().div_(n_lv - 1)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, None


def quant_general(x, bits, scale, act_sign):
  if type(bits) is int:
    return quant_int(x, bits, scale, act_sign)
  if type(bits) is str:
    return quant_fp(x, bits, scale, act_sign)
  else:
    raise Exception(
        f'Quant not defined for bits {bits} with data type {type(bits)}')


def quant_int(x, bits, scale, act_sign):

  n_lv = 2**bits

  if act_sign is False:
    x = F.hardtanh(x / scale, 0, 1)
    x = RoundQuant.apply(x, n_lv) * scale
  else:
    x = F.hardtanh(x / scale, -1, 1)
    x = RoundQuant.apply(x, torch.div(n_lv, 2, rounding_mode="floor")) * scale

  return x


def max_calib(x):
  return torch.max(x.flatten().abs())


def perc_calib(x, q=.99):
  return torch.quantile(x.abs().flatten().float(), q)


def maxf_calib(x, f=.99):
  return torch.max(x.abs().flatten().float()) * f


class einsum_linear(torch.autograd.Function):

  @staticmethod
  def forward(ctx, inpt, wgt, scale_inpt, scale_wgt, scale_err, scale_grad,
              bits_inpt, bits_wgt, bits_err, bits_grad, l_obj, act_sign, bias=None):

    # global track_stats

    ctx.bits_err = bits_err
    ctx.bits_grad = bits_grad

    ctx.l_obj = l_obj
    wgt = torch.transpose(wgt, 0, 1)

    if bits_wgt is not None:
      wgt_quant = quant_general(wgt, bits_wgt, scale_wgt, True)
      if bias is not None:
        bias_quant = quant_general(bias, bits_wgt, scale_wgt, True)
    else:
      wgt_quant = wgt
      if bias is not None:
        bias_quant = bias

    if bits_inpt is not None:
      inpt_quant = quant_general(inpt, bits_inpt, scale_inpt, act_sign)
    else:
      inpt_quant = inpt

    out = torch.einsum("bc,cd->bd", (inpt_quant, wgt_quant))

    if bias is not None:
      out += bias_quant.unsqueeze(0).expand_as(out)
    else:
      bias_quant = bias

    ctx.save_for_backward(inpt_quant, wgt_quant,
                          bias_quant, scale_err, scale_grad)

    return out

  @staticmethod
  def backward(ctx, grad_output):

    global track_stats

    if calibrate_phase:
      if ctx.l_obj.e_scale == -1:
        ctx.l_obj.e_scale = ctx.l_obj.e_calib_fn(grad_output)
      else:
        ctx.l_obj.e_scale = ctx.l_obj.e_update_stats_fn(
            ctx.l_obj.e_scale, ctx.l_obj.e_calib_fn(grad_output))

    # track e
    track_stats[ctx.l_obj.name]['e']['mean'].append(
        float(torch.mean(grad_output.flatten(), dim=0)))
    track_stats[ctx.l_obj.name]['e']['std'].append(
        float(torch.std(grad_output.flatten(), dim=0)))
    track_stats[ctx.l_obj.name]['e']['max'].append(
        float(grad_output.flatten().abs().max()))
    # track_stats[ctx.l_obj.name]['e']['calib'].append(
    #     float(ctx.l_obj.e_calib_fn(grad_output)))

    inpt_quant, wgt_quant, bias, scale_err, scale_grad = ctx.saved_tensors
    grad_input = grad_weight = None

    if ctx.bits_err is not None:
      quant_error = quant_general(
          grad_output, ctx.bits_err, scale_err, True).type(grad_output.type())

      # overflow count
      # scale_update = (grad_output.abs() > scale_err).type(grad_output.type())

      # smaller than half
      # scale_update = scale_update + - \
      #     (grad_output.abs() < scale_err / 2).type(grad_output.type())

      # update scale
      # tol = .8
      # if scale_update.mean() < 0 - ctx.l_obj.qerr_update_tol:
      #   ctx.l_obj.e_scale = ctx.l_obj.e_scale * .5
      # elif scale_update.mean() > 0 + ctx.l_obj.qerr_update_tol:
      #   ctx.l_obj.e_scale = ctx.l_obj.e_scale * 2
      # else:
      #   # no update
      #   pass

      track_stats[ctx.l_obj.name]['e']['calib'].append(scale_err)
      # # track clipping + rounding error
      # cl_err = torch.relu(torch.abs(grad_output) - scale_err)# .sum()
      # r_err = torch.abs((torch.abs(grad_output) - torch.abs(quant_error))[cl_err == 0]).sum()
      # track_stats[ctx.l_obj.name]['e']['clip_perc'].append(1 - torch.mean((cl_err == 0).float() ) )
      # track_stats[ctx.l_obj.name]['e']['clip'].append(cl_err.sum())
      # track_stats[ctx.l_obj.name]['e']['round'].append(r_err)

      # track_stats[ctx.l_obj.name]['e']['qe'].append(
      #     float(torch.mean((quant_error - grad_output)**2)))
      # track_stats[ctx.l_obj.name]['e']['bs'].append(inpt_quant.shape[0])
    else:
      quant_error = grad_output
      # track_stats[ctx.l_obj.name]['e']['qe'].append(0)
      # track_stats[ctx.l_obj.name]['e']['bs'].append(inpt_quant.shape[0])

    if ctx.needs_input_grad[0]:
      # propagate quantized error
      grad_input = torch.einsum(
          "bc,dc->bd", (quant_error, wgt_quant.type(quant_error.type())))
    if ctx.needs_input_grad[1]:
      grad_weight = torch.einsum(
          "bc,bd->dc", (quant_error, inpt_quant.type(quant_error.type())))
      if ctx.bits_grad is not None:
        grad_weight = quant_general(
            grad_weight, ctx.bits_grad, scale_grad, True)

    if bias is not None and ctx.needs_input_grad[2]:
      grad_bias = quant_error.sum(0).squeeze(0)

    if calibrate_phase:
      if ctx.l_obj.g_scale == -1:
        ctx.l_obj.g_scale = ctx.l_obj.g_calib_fn(grad_weight)
        ctx.l_obj.g_bias = torch.mean(grad_weight)
      else:
        ctx.l_obj.g_scale = ctx.l_obj.g_update_stats_fn(
            ctx.l_obj.g_scale, ctx.l_obj.g_calib_fn(grad_weight))
        ctx.l_obj.g_bias = .95 * ctx.l_obj.g_bias + \
            (1 - .95) * torch.mean(grad_weight)

    return grad_input, torch.transpose(grad_weight, 0, 1), None, None, None, \
        None, None, None, None, None, None, None, None, None, None, None, grad_bias


class QLinear(nn.Module):
  # based off
  # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
  __constants__ = ['in_features', 'out_features']
  in_features: int
  out_features: int
  weight: torch.Tensor

  def __init__(self, in_features: int, out_features: int, bias: bool = False,
               device=None, dtype=None, a_bits: int = 0, w_bits: int = 0,
               e_bits: int = 0, g_bits: int = 0,
               a_calib_fn: Callable = perc_calib,
               w_calib_fn: Callable = perc_calib,
               e_calib_fn: Callable = perc_calib,
               g_calib_fn: Callable = perc_calib,
               a_calib_mom: float = .95,
               w_calib_mom: float = .95,
               e_calib_mom: float = .95,
               g_calib_mom: float = .95, name='', qerr_update_tol=0., act_sign=True) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(QLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty(
        (out_features, in_features), **factory_kwargs))
    self.qweight = nn.Parameter(torch.empty(
        (out_features, in_features), **factory_kwargs))
    if bias:
      self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

    self.name = name

    self.qerr_update_tol = qerr_update_tol

    self.act_sign = act_sign

    self.a_bits = a_bits
    self.w_bits = w_bits
    self.e_bits = e_bits
    self.g_bits = g_bits

    self.a_calib_fn = a_calib_fn
    self.w_calib_fn = w_calib_fn
    self.e_calib_fn = e_calib_fn
    self.g_calib_fn = g_calib_fn

    self.a_calib_mom = a_calib_mom
    self.w_calib_mom = w_calib_mom
    self.e_calib_mom = e_calib_mom
    self.g_calib_mom = g_calib_mom

    if torch.cuda.is_available():
      device = torch.device("cuda:0")
    else:
      device = torch.device("cpu")

    self.a_scale = torch.tensor([-1], device=device)
    self.w_scale = torch.tensor([-1], device=device)
    self.e_scale = torch.tensor([-1], device=device)
    self.g_scale = torch.tensor([-1], device=device)

    self.a_bias = torch.tensor([-1], device=device)
    self.w_bias = torch.tensor([-1], device=device)
    self.e_bias = torch.tensor([-1], device=device)
    self.g_bias = torch.tensor([-1], device=device)

    if a_calib_mom == 'max':
      self.a_update_stats_fn = lambda x, y: torch.maximum(x, y)
    else:
      self.a_update_stats_fn = lambda x, y: a_calib_mom * \
          x + (1 - a_calib_mom) * y

    if w_calib_mom == 'max':
      self.w_update_stats_fn = lambda x, y: torch.maximum(x, y)
    else:
      self.w_update_stats_fn = lambda x, y: w_calib_mom * \
          x + (1 - w_calib_mom) * y

    if e_calib_mom == 'max':
      self.e_update_stats_fn = lambda x, y: torch.maximum(x, y)
    else:
      self.e_update_stats_fn = lambda x, y: e_calib_mom * \
          x + (1 - e_calib_mom) * y

    if g_calib_mom == 'max':
      self.g_update_stats_fn = lambda x, y: torch.maximum(x, y)
    else:
      self.g_update_stats_fn = lambda x, y: g_calib_mom * \
          x + (1 - g_calib_mom) * y

    global track_stats

    if name != 'classifier':
      track_stats[name] = {
          'a': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': []},
          'w': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': []},
          'e': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': [],
                'bs': [], 'clip': [], 'round': [], 'clip_perc': []},
          'g': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': []},
      }
    else:
      if 'classifier' not in track_stats.keys():
        track_stats[name] = {
            'a': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': []},
            'w': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': []},
            'e': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': [],
                  'bs': [], 'clip': [], 'round': [], 'clip_perc': []},
            'g': {'mean': [], 'std': [], 'max': [], 'qe': [], 'calib': []},
        }

  def reset_parameters(self) -> None:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if calibrate_phase:
      with torch.no_grad():
        if self.a_scale == -1:
          self.a_scale = self.a_calib_fn(input)
          self.a_bias = torch.mean(input)
        else:
          self.a_scale = self.a_update_stats_fn(
              self.a_scale, self.a_calib_fn(input))
          self.a_bias = .95 * self.a_bias + (1 - .95) * torch.mean(input)

        if self.w_scale == -1:
          self.w_scale = self.w_calib_fn(self.weight)
          self.w_bias = torch.mean(self.weight)
        else:
          self.w_scale = self.w_update_stats_fn(
              self.w_scale, self.w_calib_fn(self.weight))
          self.w_bias = .95 * self.w_bias + (1 - .95) * torch.mean(self.weight)

      return einsum_linear.apply(input, self.weight, torch.tensor([1.],
                                 device=input.device), torch.tensor([1.],
                                 device=input.device), torch.tensor([1.],
                                 device=input.device), torch.tensor([1.],
                                 device=input.device), None, None,
                                 None, None, self, None, None,
                                 None, None, self.act_sign, self.bias,)

    return einsum_linear.apply(input, self.weight, self.a_scale,
                               self.w_scale, self.e_scale,
                               self.g_scale, self.a_bits, self.w_bits,
                               self.e_bits, self.g_bits, self, self.a_bias,
                               self.w_bias, self.e_bias,
                               self.g_bias, self.act_sign, self.bias, )

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None
    )


def str_to_calib_fn(x):
  if x == max:
    return max_calib
  elif 'perc' in x:
    return functools.partial(perc_calib, q=float(x.split('_')[1]))
  elif 'maxf' in x:
    return functools.partial(maxf_calib, f=float(x.split('_')[1]))
  else:
    raise Exception('Unknown calibration argument passed: ' + x)


@dataclasses.dataclass
class FloatingPointBounds:
  # orginally copied from
  # https://github.com/google/aqt/blob/main/aqt/jax_legacy/jax/fp_cast.py
  """Dataclass representing the bounds for a floating-point type.
  The type is presumed to have 'flush to zero' semantics.
  Attributes:
    flush_to_zero_bound: The magnitude of the smallest representable value. If
      a logical value with an absolute value less than this is cast to this
      type, it is flushed to zero.
    saturation_bound: The magnitude of the largest representable value. If a
      logical value with an absolute value greater than this is cast to this
      type, it is clipped to this value.
  """

  flush_to_zero_bound: float
  saturation_bound: float


def get_bounds(exp_min, exp_max, sig_bits):
  # orginally copied from
  # https://github.com/google/aqt/blob/main/aqt/jax_legacy/jax/fp_cast.py
  """Returns the clipping bounds for a giving floating-point specification.
  Args:
    exp_min: The denormal exponent of the target format.
    exp_max: Maximum exponent of the target format (no support for infs & nans)
    sig_bits: The number of significant bits in the target format (excluding
    the hidden bit).
  Returns:
    A FloatingPointBounds dataclass.
  """
  return FloatingPointBounds(
      flush_to_zero_bound=2**exp_min,
      saturation_bound=2**exp_max * (2 - 2**-sig_bits))


class RoundFPQuant(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, exp_min, exp_max, sig_bits, base=2):
    # orginally copied from
    # https://github.com/google/aqt/blob/main/aqt/jax_legacy/jax/fp_cast.py
    """Downcast bfloat16, or float32 to a lower precision floating-point.
    - The downcast returns an argument of the *same* type as x but with
      numerical characteristics of the target fp-format.
    - The implementation (sat)urates to the largest representable number in the
      range instead of rounding to infinity.
    - Rounding mode is round-nearest-even for TPU (or configured rounding).
    - Support for special values:
      - The target fp is assumed to *not* support special values, as exp_max is
        used for the numerical range. For the sake of emulation special values
        are propagated as values on the input type, such that
        downcast(nan/inf) -> nan/inf is preserved.
    - Denormals: the target format doesn't support denormals. The minimum
      representable positive value is 2^exp_min. Denormals flush to zero (ftz).
    Args:
      x: The argument to be converted.
      exp_min: The denormal exponent of the target format.
      exp_max: Maximum exponent of the target format (no support for infs &
        nans)
      sig_bits: The number of significant bits in the target format (excluding
        the hidden bit).
    Returns:
     Cast of x to a lower precision float, emulating degraded precision, but of
     the same type as x.
    """

    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
      raise ValueError('Argument is expected to be of type torch.float32 '
                       'or torch.bfloat16')

    # Mask for exponent bits in fp32 representation.
    if base == 2:
      exp_mask = 0x7f800000
      specials_bound = 0x7f800000
    elif base == 4:
      raise Exception('Base 4 quantization not yet implemented.')
      exp_mask = 0x2A800000
      specials_bound = 0x2A800000
    else:
      raise Exception('Unsupported base type for fp: ' + str(base))
    # NaNs / Infs have representation-value >= specials_bound.

    # Binary representation of +1.0.
    one = 0x3f800000

    # Mask for mantissa bits (lower 23-bits) of fp32 representation.
    mant_mask = 0x007fffff

    xf = x.to(torch.float32)
    xi = xf.view(torch.int32)

    exp = xi & exp_mask
    # Scale the argument to the unit binade.
    xi_one = (xi & mant_mask) | one
    offset = 2**(23 - sig_bits)
    # Addition of offset creates alignment to shift-off and round trailing bits
    # Subtraction brings the rounded result back to the unit binade.
    xf_one_rnd = (xi_one.view(torch.float32) + offset) - offset

    # Scale back to the original binade.
    xf_rnd = xf_one_rnd * exp.view(torch.float32)
    bounds = get_bounds(exp_min=exp_min, exp_max=exp_max, sig_bits=sig_bits)
    xf_rnd_sat = torch.minimum(xf_rnd, torch.tensor(bounds.saturation_bound))

    # Flush denormals to zero and recover sign.
    xf_rnd_sat_ftz = torch.sign(xf) * xf_rnd_sat * \
        (xf_rnd_sat >= bounds.flush_to_zero_bound)
    xf_rnd_sat_ftz = torch.where(exp >= specials_bound, xf, xf_rnd_sat_ftz)
    return xf_rnd_sat_ftz.to(x.dtype)

  @staticmethod
  def backward(ctx, grad_output):
    # STE
    return grad_output, None, None, None


def ibm(x, bits=3):
  x = x * 1.6
  # logarithm of x base b = log(x)/log(b)
  ebit = torch.floor(torch.log(torch.abs(x))
                     / torch.log(torch.tensor([4.], device=x.device)))
  return torch.where(ebit < -bits, 0., torch.where(ebit >= bits, torch.sign(x) * 4**bits, torch.sign(x) * 4**ebit))


def ibm8(x, bits=3):
  x = x * 1.6
  # logarithm of x base b = log(x)/log(b)
  ebit = torch.floor(torch.log(torch.abs(x))
                     / torch.log(torch.tensor([8.], device=x.device)))
  return torch.where(ebit < -bits, 0., torch.where(ebit >= bits, torch.sign(x) * 8**bits, torch.sign(x) * 8**ebit))


def ibm16(x, bits=3):
  x = x * 1.6
  # logarithm of x base b = log(x)/log(b)
  ebit = torch.floor(torch.log(torch.abs(x))
                     / torch.log(torch.tensor([16.], device=x.device)))
  return torch.where(ebit < -bits, 0., torch.where(ebit >= bits, torch.sign(x) * 16**bits, torch.sign(x) * 16**ebit))


def quant_fp(x, bits, scale, base=2, asym=False):
  if bits == 'four':
    # trying to implement base four
    max_val = 64
    x = F.hardtanh(x * 1 / scale, -1, 1) * max_val
    x = ibm(x)
    x = x * 1 / max_val * scale

    return x

  if bits == 'eight':
    # trying to implement base four
    max_val = 512
    x = F.hardtanh(x * 1 / scale, -1, 1) * max_val
    x = ibm8(x)
    x = x * 1 / max_val * scale

    return x

  if bits == 'sixt':
    # trying to implement base four
    max_val = 4096
    x = F.hardtanh(x * 1 / scale, -1, 1) * max_val
    x = ibm8(x)
    x = x * 1 / max_val * scale

    return x

  exp_bits, man_bits = [int(num) for num in bits.split("e")[1].split("m")]

  exp_max = base ** (exp_bits - 1)
  exp_min = -base ** (exp_bits - 1) + 1

  max_val = 2**exp_max * (2 - 2**-man_bits)

  # remove last representation in favor of symmetric around zero
  if not asym:
    max_val = max_val - (2**(-1 * exp_min)) * 2**-(man_bits - 1)

  if max_val == 0:
    max_val = 2**(exp_bits + 1)
    max_val = max_val - (2**(-1 * exp_min - 1)) * 2**-(man_bits - 1)

  if max_val > 65_504 * 2:  # I think this is enough to just raw work then.
    # clip
    x = F.hardtanh(x * 1 / scale, -1, 1) * scale

    # round
    x = RoundFPQuant.apply(x, exp_min, exp_max, man_bits, base)
  else:
    # clip and scale to range
    x = F.hardtanh(x * 1 / scale, -1, 1) * max_val
    if asym:
      x = torch.clamp(
          x, min=-(max_val - (2**(-1 * exp_min - 1)) * 2**-(man_bits - 1)))
      # x = torch.clamp(x, max = max_val - (2**(-1 * exp_min-1)) * 2**-(man_bits - 1))

    # round
    x = RoundFPQuant.apply(x, exp_min, exp_max, man_bits, base)

    # scale back
    x = x * 1 / max_val * scale

  return x
