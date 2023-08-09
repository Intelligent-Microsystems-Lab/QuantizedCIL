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

from hadamard import hadamard, prime_factors, make_hadamard

track_stats = {'grads': {}, 'acts': {}, 'wgts': {}, 'grad_stats':{}, 'test_acc':[], 'train_acc':[], 'loss':[]}
calibrate_phase = False
quantizeFwd = False
quantizeBwd = False
quantGradRound = "standard"
quantCalibrate = "max"
quantTrack = False
quantBits = 4
quantMethod = "luq"
quantBatchSize = 128

quantGradMxScale = 1.

scale_library = {'a': {}, 'w': {}, 'g': {}}

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

  obj.quantizeFwd = quantizeFwd
  obj.quantizeBwd = quantizeBwd

  obj.c1 = 12.1
  obj.c2 = 12.2
  obj.stochastic = quantGradRound
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
          track_stats['grad_stats'][c_path + '_' + attr_str] = {'max':[], 'min':[],'norm':[], 'mean':[]}
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
        track_stats['grad_stats'][c_path + '_' + attr_str] = {'max':[], 'min':[],'norm':[], 'mean':[]}
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
        if quantMethod == 'luq':
          tmp_meth = Linear_LUQ
        elif quantMethod == 'ours':
          tmp_meth = Linear_Ours
        else:
          raise Exception('Unknown quant method: ' + quantMethod)
        setattr(m, attr_str, tmp_meth(in_features=target_attr.in_features,
                                        out_features=target_attr.out_features,
                                        bias=hasattr(target_attr, 'bias'),
                                        uname=c_path + '_' + attr_str,))
        if lin_w is not None:
          m.fc.weight = nn.Parameter(lin_w)
        if lin_b is not None:
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
    track_stats['grad_stats'][ctx.name]['norm'].append(torch.linalg.vector_norm(grad_output.flatten()).cpu())

    # grad max
    track_stats['grad_stats'][ctx.name]['max'].append(grad_output.max().cpu())

    # grad min
    track_stats['grad_stats'][ctx.name]['min'].append(grad_output.min().cpu())

    # grad mean
    track_stats['grad_stats'][ctx.name]['mean'].append(grad_output.mean().cpu())


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

    if self.quantizeFwd:
      
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

    if self.quantizeFwd:
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
        alpha = mx / 2**(2**bits - 2) # was 1 before, need to be 2 centered around 0

        if quantGradRound == "stochastic":
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

        if quantGradRound == "stochastic":
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




class FLinearQ(torch.autograd.Function):
  generate_vmap_rule = True

  @staticmethod
  def forward(x, w, h_out, h_bs):
    # output = x @ w
    output = F.linear(x, w)
    return output

  @staticmethod
  def setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3])


  @staticmethod
  def backward(ctx, grad_output):
    x, w, h_out, h_bs = ctx.saved_tensors
    # w = w.T

    # if quant:

    w_h1 = h_out @ w
    grad_output_h1 = grad_output @ h_out

    # quant grad_output

    grad_input = (grad_output_h1 @ w_h1) * 1/prime_factors(h_out.shape[0])

    x_h2 = h_bs @ x
    grad_output_h2 = grad_output.T @ h_bs

    # quant grad_output

    grad_w = (grad_output_h2 @ x_h2) * 1/prime_factors(h_bs.shape[0])

    # np.testing.assert_allclose(grad_w.cpu(), (grad_output.T @ x).cpu())
    # np.testing.assert_allclose(grad_input.cpu(), (grad_output @ w).cpu() )
      
    # else:
    #   grad_input = grad_output @ w

    #   grad_w = grad_output.T @ x

    return grad_input, grad_w, None, None, None, None


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


def lsq(x, s, Qn, Qp):
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

    self.register_buffer('hadamard_out', torch.tensor(make_hadamard(self.out_features), dtype = self.weight.dtype))
    self.register_buffer('hadamard_bs', torch.tensor(make_hadamard(quantBatchSize), dtype = self.weight.dtype))

    self.lsq_act = Parameter(torch.tensor([1.], dtype=torch.float32))
    self.lsq_wgt = Parameter(torch.tensor([ self.weight.abs().mean() * 2 / np.sqrt(self.QpW) ], dtype=torch.float32))


  def forward(self, input):

    if self.quantizeFwd:
      # w_q = UniformQuantizeSawb.apply(
      #       self.weight, self.c1, self.c2, self.QpW, self.QnW)

      # w_q = lsq(self.weight, self.lsq_wgt, self.QnW, self.QpW)

      if torch.min(input) < 0:
        self.QnA = -2 ** (self.abits - 1) + 1
        self.QpA = 2 ** (self.wbits - 1) - 1

      
      # qinput = UniformQuantizeSawb.apply(
      #       input, self.c1, self.c2, self.QpA, self.QnA)
      # qinput = lsq(input, self.lsq_act, self.QnA, self.QpA)

      w_q = self.weight
      qinput = input

      # TODO: optimize speed of hadamard creation
      if input.shape[0] != quantBatchSize:
        # biggest_pow2 = prime_factors(input.shape[0])
        h_bs = torch.tensor(make_hadamard(input.shape[0]), dtype=self.weight.dtype).to(self.weight.device)
      else:
        h_bs = self.hadamard_bs

      output = FLinearQ.apply(qinput, w_q, self.hadamard_out, h_bs) + self.bias

      # LUQ bwd
      # output = GradStochasticClippingQ.apply(
      #   output, self.quantizeBwd, self.layerIdx, self.repeatBwd, self.uname)

    else:
      output = F.linear(input, self.weight, self.bias)

    return {'logits': output}


class Conv2d_Ours(nn.Conv2d):

  """docstring for Conv2d_BF16."""

  def __init__(self, uname, *args, **kwargs):
    super(Conv2d_Ours, self).__init__(*args, **kwargs)
    init_properties(self, uname)
 
    self.register_buffer('hadamard_out', torch.tensor(make_hadamard(self.out_channels), dtype = self.weight.dtype))
    self.register_buffer('hadamard_bs', torch.tensor(torch.zeros((1,1)), dtype = self.weight.dtype))

    self.lsq_act = Parameter(torch.tensor([1.], dtype=torch.float32))
    self.lsq_wgt = Parameter(torch.tensor([ self.weight.abs().mean() * 2 / np.sqrt(self.QpW) ], dtype=torch.float32))


  def forward(self, input):

    if self.quantizeFwd:
      # w_q = UniformQuantizeSawb.apply(
      #       self.weight, self.c1, self.c2, self.QpW, self.QnW)

      # w_q = lsq(self.weight, self.lsq_wgt, self.QnW, self.QpW)

      # if torch.min(input) < 0:
      #   self.QnA = -2 ** (self.abits - 1) + 1
      #   self.QpA = 2 ** (self.wbits - 1) - 1

      # qinput = UniformQuantizeSawb.apply(
      #       input, self.c1, self.c2, self.QpA, self.QnA)
      # qinput = lsq(input, self.lsq_act, self.QnA, self.QpA)

      w_q = self.weight
      qinput = input

      # TODO: optimize speed of hadamard creation
      
        

      qinput = torch.nn.functional.unfold(qinput, self.kernel_size, padding=self.padding, stride=self.stride).transpose(1, 2)
      w_q = w_q.view(w_q.size(0), -1).t()


      if self.hadamard_bs.sum() == 0:
        self.hadamard_bs = torch.tensor(make_hadamard(qinput.shape[1]), dtype = self.weight.dtype).to(self.weight.device)

      
      flinearq_fn = torch.vmap(FLinearQ.apply)
      out = flinearq_fn(qinput, w_q.T.unsqueeze(0).repeat(qinput.shape[0], 1, 1), self.hadamard_out.unsqueeze(0).repeat(qinput.shape[0], 1, 1), self.hadamard_bs.unsqueeze(0).repeat(qinput.shape[0], 1, 1))

      # out = FLinearQ.apply(qinput, w_q, self.hadamard_out, h_bs,)
      # import pdb; pdb.set_trace()
      # 

      out = out.transpose(1, 2)
      output = out.view((input.shape[0], self.out_channels, int(input.shape[-2]/self.stride[0]), int(input.shape[-1]/self.stride[1]))) + self.bias.view(1,-1,1,1)

      # np.testing.assert_allclose(output_cmp.cpu().detach().numpy(), output.cpu().detach().numpy(), rtol=1e-05, atol=1e-2)

      # output = FLinearQ.apply(qinput, w_q, self.quantizeBwd, self.uname, self.hadamard_out, h_bs,) + self.bias

      # LUQ bwd
      # output = GradStochasticClippingQ.apply(
      #   output, self.quantizeBwd, self.layerIdx, self.repeatBwd, self.uname)

    else:
      output = F.linear(input, self.weight, self.bias)

    return output
