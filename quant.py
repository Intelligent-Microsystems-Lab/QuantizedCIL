# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer and Martin Schiemer
# Quantized training.


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.autograd.function import Function
from torch.autograd.function import InplaceFunction

from backbones.linears import SimpleLinear

from squant_function import SQuant_func

track_stats = {'grads': {}, 'acts': {}, 'wgts': {}, 'grad_stats':{}, 'test_acc':[], 'train_acc':[], 'loss':[]}
calibrate_phase = False
quantizeFwd = False
quantizeBwd = False
quantGradRound = "standard"
quantCalibrate = "max"
quantTrack = False
quantBits = 4

quantGradMxScale = 1.

scale_library = {'a': {}, 'w': {}, 'g': {}}



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
          setattr(m, attr_str,
                  Conv2d_LUQ(in_channels=target_attr.in_channels,
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
        setattr(m, attr_str, Linear_LUQ(in_features=target_attr.in_features,
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

      # import pdb; pdb.set_trace()
      output.div_(scale)
      output.clamp_(Qn, Qp).round_()
      output.mul_(scale)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    # straight-through estimator
    grad_input = grad_output
    return grad_input, None, None, None, None


class UniformQuantizeSawb_calib(InplaceFunction):

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
    self.abits = quantBits
    self.wbits = quantBits

    self.QnW = -2 ** (self.wbits - 1)
    self.QpW = 2 ** (self.wbits - 1)
    self.QnA = 0
    self.QpA = 2 ** self.abits - 1

    self.register_buffer('init_stateW', torch.zeros(1))
    self.register_buffer('init_stateA', torch.zeros(1))

    self.register_buffer('gradScaleW', torch.zeros(1))
    self.register_buffer('gradScaleA', torch.zeros(1))

    self.quantizeFwd = quantizeFwd
    self.quantizeBwd = quantizeBwd

    self.c1 = 12.1
    self.c2 = 12.2
    self.stochastic = quantGradRound
    self.calibrate = quantCalibrate
    self.repeatBwd = 1

    self.uname = uname

  def forward(self, input):

    if self.quantizeFwd:
      if False: # self.calibrate: TODO: fix
        if calibrate_phase:
          with torch.no_grad():
            if self.uname not in scale_library['w']:
              scale_library['w'][self.uname] = {}
              scale_library['w'][self.uname]['clip_w'] = (
                  self.c1 * torch.sqrt(torch.mean(self.weight**2))
              ) - (self.c2 * torch.mean(self.weight.abs()))
              scale_library['w'][self.uname]['scale_w'] = 2 * \
                  scale_library['w'][self.uname]['clip_w'] / \
                  (self.QpW - self.QnW)
            else:
              scale_library['w'][self.uname]['clip_w'
                                             ] = scale_library['w'][
                  self.uname]['clip_w'] * .9 + .1 * (
                  (self.c1 * torch.sqrt(torch.mean(self.weight**2))
                   ) - (self.c2 * torch.mean(self.weight.abs())))
              scale_library['w'][
                  self.uname]['scale_w'] = scale_library['w'][
                  self.uname]['scale_w'] * .9 + .1 * (
                  2 * scale_library['w'][self.uname]['clip_w']
                  / (self.QpW - self.QnW))
          w_q = self.weight
        else:
          w_q = UniformQuantizeSawb_calib.apply(
              self.weight, self.QpW, self.QnW, scale_library['w'][
                  self.uname]['scale_w'], scale_library['w'][
                  self.uname]['clip_w'])
      else:
        w_q = UniformQuantizeSawb.apply(
            self.weight, self.c1, self.c2, self.QpW, self.QnW)

      if torch.min(input) < 0:
        self.QnA = -2 ** (self.abits - 1)

      if False: # self.calibrate: TODO: fix
        if calibrate_phase:
          with torch.no_grad():
            if self.uname not in scale_library['a']:
              scale_library['a'][self.uname] = {}
              scale_library['a'][self.uname]['clip_a'] = (
                  self.c1 * torch.sqrt(torch.mean(input**2))
              ) - (self.c2 * torch.mean(input.abs()))
              scale_library['a'][self.uname]['scale_a'] = 2 * \
                  scale_library['a'][self.uname]['clip_a'] / \
                  (self.QpA - self.QnA)
            else:
              scale_library['a'][self.uname]['clip_a'
                                             ] = scale_library['a'][
                  self.uname]['clip_a'] * .9 + .1 * (
                  (self.c1 * torch.sqrt(torch.mean(input**2))
                   ) - (self.c2 * torch.mean(input.abs())))
              scale_library['a'][self.uname]['scale_a'
                                             ] = scale_library['a'][
                  self.uname]['scale_a'] * .9 + .1 * (
                  2 * scale_library['a'][self.uname]['clip_a']
                  / (self.QpA - self.QnA))
          qinput = input
        else:
          qinput = UniformQuantizeSawb_calib.apply(
              input, self.QpA, self.QnA,
              scale_library['a'][self.uname]['scale_a'],
              scale_library['a'][self.uname]['clip_a'])
      else:
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
    self.fullName = ''
    self.statistics = []
    self.layerIdx = 0

    self.alpha = Parameter(torch.tensor([1], dtype=torch.float32))
    self.beta = Parameter(torch.tensor([1], dtype=torch.float32))
    self.abits = quantBits
    self.wbits = quantBits

    self.QnW = -2 ** (self.wbits - 1)
    self.QpW = 2 ** (self.wbits - 1)
    self.QnA = 0
    self.QpA = 2 ** self.abits - 1

    self.register_buffer('init_stateW', torch.zeros(1))
    self.register_buffer('init_stateA', torch.zeros(1))

    self.register_buffer('gradScaleW', torch.zeros(1))
    self.register_buffer('gradScaleA', torch.zeros(1))

    self.quantizeFwd = quantizeFwd
    self.quantizeBwd = quantizeBwd

    self.c1 = 12.1
    self.c2 = 12.2
    self.stochastic = quantGradRound
    self.calibrate = quantCalibrate
    self.repeatBwd = 1

    self.uname = uname

  def forward(self, input):

    if self.quantizeFwd:
      if False: #self.calibrate: TODO fix
        if calibrate_phase:
          with torch.no_grad():
            if self.uname not in scale_library['w']:
              scale_library['w'][self.uname] = {}
              scale_library['w'][self.uname]['clip_w'] = (
                  self.c1 * torch.sqrt(torch.mean(self.weight**2))
              ) - (self.c2 * torch.mean(self.weight.abs()))
              scale_library['w'][self.uname]['scale_w'] = 2 * \
                  scale_library['w'][self.uname]['clip_w'] / \
                  (self.QpW - self.QnW)
            else:
              scale_library['w'][self.uname]['clip_w'
                                             ] = scale_library['w'][
                  self.uname
              ]['clip_w'] * .9 + .1 * (
                  (self.c1 * torch.sqrt(torch.mean(self.weight**2))
                   ) - (self.c2 * torch.mean(self.weight.abs())))
              scale_library['w'][self.uname]['scale_w'] = scale_library['w'][
                  self.uname]['scale_w'] * .9 + .1 * (
                  2 * scale_library['w'][self.uname]['clip_w'
                                                     ] / (self.QpW - self.QnW))
          w_q = self.weight
        else:
          w_q = UniformQuantizeSawb_calib.apply(
              self.weight, self.QpW, self.QnW, scale_library['w'][
                  self.uname]['scale_w'], scale_library['w'][
                  self.uname]['clip_w'])
      else:
        w_q = UniformQuantizeSawb.apply(
            self.weight, self.c1, self.c2, self.QpW, self.QnW)

      if torch.min(input) < 0:
        self.QnA = -2 ** (self.abits - 1)

      if False: # self.calibrate: TODO fix
        if calibrate_phase:
          with torch.no_grad():
            if self.uname not in scale_library['a']:
              scale_library['a'][self.uname] = {}
              scale_library['a'][self.uname]['clip_a'] = (
                  self.c1 * torch.sqrt(torch.mean(input**2))
              ) - (self.c2 * torch.mean(input.abs()))
              scale_library['a'][self.uname]['scale_a'] = 2 * \
                  scale_library['a'][self.uname]['clip_a'] / \
                  (self.QpA - self.QnA)
            else:
              scale_library['a'][self.uname]['clip_a'
                                             ] = scale_library['a'][
                  self.uname]['clip_a'] * .9 + .1 * (
                  (self.c1 * torch.sqrt(torch.mean(input**2))
                   ) - (self.c2 * torch.mean(input.abs())))
              scale_library['a'][self.uname]['scale_a'
                                             ] = scale_library['a'][
                  self.uname]['scale_a'] * .9 + .1 * (
                  2 * scale_library['a'][self.uname]['clip_a'
                                                     ] / (self.QpA - self.QnA))
          qinput = input
        else:
          qinput = UniformQuantizeSawb_calib.apply(
              input, self.QpA, self.QnA, scale_library['a'][self.uname][
                  'scale_a'], scale_library['a'][self.uname]['clip_a'])
      else:
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
