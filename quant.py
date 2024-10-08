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

from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear, reduce_proxies

from utils.data_manager import DummyDataset
from torch.utils.data import DataLoader

from luq import LUQ
from luq.LUQ import Conv2d_LUQ, Linear_LUQ

try:
  import lptorch as lp
  qnn = lp.nn
except:
  # makes code executable without lptorch compiled
  pass

from hadamard import make_hadamard, biggest_power2_factor


EPSILON = 1e-32

track_stats = {'grads': {}, 'acts': {}, 'wgts': {},
               'grad_stats': {}, 'test_acc': [], 'train_acc': [], 'loss': [],
               'zeros': {}, 'maxv':{}}
grad_track_batch = []
loss_track_batch = []
grad_track_epoch = []
loss_track_epoch = []
batches_per_task = []
calibrate_phase = False
quantizeFwd = False
quantizeBwd = False
quantCalibrate = "max"
quantTrack = False
quantBits = 4
quantAccBits = 8
quantAccFWD = False
quantAccBWD = False
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
quantHadOff = False
quantRequantize = True
quantRelevantMeasurePass = False
quantUpdateScalePhase = False
quantUpdateLowThr = .7
quantUpdateHighThr = .3
global_args = None
quant_bin_use_perc = []
quant_bin_use_hist = []

current_uname = ''

QpW = None
QnW = None
QpA = None
QnA = None

quantGradMxScale = 1.

quantFP134_rep = '555555543210'

epochnr = 0
batchnr = 0
gradient_library = {}
scale_library = {}
scale_library_hist = []
quant_range_use_perc = {}
weight_recording = {}


# exponent_bits = 2 # fp4
# mantissa_bits = 1 # fp4
exponent_bits = 5 # fp8
mantissa_bits = 2 # fp8
exponent_bits_acc = 5 # fp8
mantissa_bits_acc = 2 # fp8
# exponent_bits = 5 # fp16
# mantissa_bits = 10 # fp16
# exponent_bits_acc = 5 # shehab
# mantissa_bits_acc = 12 # shehab


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

          p.data -= self.layer_lr[p] * p.grad.data

          def dynamic_intQ2(x, scale=None):
            # can only be used in BWD - not differentiable
            # torch.quantile(x.abs(), .99) # TODO optimize calibration
            if scale is None:
              scale = 1 # global_args["quantile"]
            mx = x.abs().max() * scale
            scale = mx / (2**(quantWgtStoreBits - 1) - 1)
            x = torch.clamp(x, -mx, mx)
            # epsilion for vmap # TODO eps size?
            return torch.round(x / (scale + EPSILON)) * scale

          p.data = dynamic_intQ2(p.data)


def init_properties(obj, uname):
  obj.fullName = ''
  obj.statistics = []
  obj.layerIdx = 0

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

def set_lptorch_quant():
  global quantMethod
  if quantMethod == 'fp130_4':
    df_format = [int(x) for x in list('11111110')] #  [int(x) for x in list(quantFP134_rep)]#$
  else:
    df_format = [int(x) for x in list('555555543210')] #  [int(x) for x in list(quantFP134_rep)]#$
  n_format = [int(x) for x in list(quantFP134_rep)]
  lp.set_activ_quant(lp.quant.quant(lp.quant.custom_fp_format(df_format), room=1, stochastic=False))
  lp.set_error_quant(lp.quant.quant(lp.quant.custom_fp_format(df_format), room=1, stochastic=True))
  lp.set_weight_quant(lp.quant.quant(lp.quant.custom_fp_format(n_format), room=0, ch_wise=True))
  lp.set_grad_quant(lp.quant.quant(lp.quant.custom_fp_format(df_format), room=2, stochastic=True))
  lp.set_master_quant(lp.quant.quant(lp.quant.fp_format(exp_bit=6, man_bit=9), stochastic=True))
  lp.set_hysteresis_update(True)


def Conv_FP134(in_channels, out_channels,kernel_size, stride, padding, 
                  padding_mode, dilation, groups, bias, uname):
  set_lptorch_quant()
  return qnn.QLayer(nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           padding_mode=padding_mode,
                           dilation=dilation,
                           groups=groups,
                           bias=bias), last=True )

def Linear_FP134(in_features, out_features, bias, uname):
  set_lptorch_quant()
  return qnn.QLayer(nn.Linear(in_features = in_features, out_features = out_features, bias = bias), last=True, ret_dict = True)

def place_quant(m, lin_w, lin_b, c_path='',is_fc_layer=False,debug=False):
  if debug == True:
    import pdb; pdb.set_trace()

  if quantMethod == 'noq':
    if is_fc_layer:
      return m
    else:
      return

  try:
    if isinstance(m, qnn.QLayer):
      return
  except:
    # makes code executable without lptorch compiled
    pass

  if is_fc_layer:
    if quantMethod == 'luq_og' or quantMethod == 'luq_corrected':
      if quantMethod == 'luq_corrected':
        LUQ.corrected_version = True
      tmp_meth = Linear_LUQ
    elif quantMethod == 'ours':
      #TODO fix cosine linear for LUQ ours and fp134
      # if isinstance(target_attr, CosineLinear) and False:
      #   tmp_meth = CosineLinear_Ours
      # else:
      tmp_meth = Linear_Ours
    elif 'fp13' in quantMethod:
      tmp_meth = Linear_FP134
    else:
      raise Exception('Unknown quant method: ' + quantMethod)

    return tmp_meth(in_features=m.in_features,
                                        out_features=m.out_features,
                                        bias=getattr(
                                            m, 'bias') is not None,
                                        uname=c_path + '_fc',)

  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      # if isinstance(target_attr, nn.Conv2d):
      #   import pdb; pdb.set_trace()
      if isinstance(target_attr, nn.Conv2d):
        if not hasattr(target_attr, 'c1'):
          if quantMethod == 'luq_og' or quantMethod == 'luq_corrected':
            if quantMethod == 'luq_corrected':
              LUQ.corrected_version = True
            tmp_meth = Conv2d_LUQ
          elif quantMethod == 'ours':
            tmp_meth = Conv2d_Ours
          elif 'fp13' in quantMethod:
            tmp_meth = Conv_FP134
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
      if isinstance(target_attr,
                    nn.Linear) or isinstance(target_attr,
                                             SimpleLinear) or isinstance(target_attr,
                                                                         CosineLinear):
        if not hasattr(target_attr, 'c1'):
          if quantMethod == 'luq_og' or quantMethod == 'luq_corrected':
            if quantMethod == 'luq_corrected':
              LUQ.corrected_version = True
            tmp_meth = Linear_LUQ
          elif quantMethod == 'ours':
            #TODO fix cosine linear for LUQ ours and fp134
            if isinstance(target_attr, CosineLinear) and False:
              tmp_meth = CosineLinear_Ours
            else:
              tmp_meth = Linear_Ours
          elif 'fp13' in quantMethod:
            tmp_meth = Linear_FP134
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
              if 'fp13' in quantMethod:
                m.fc.module.weight.data = lin_w
              else:
                m.fc.weight.data = lin_w
          if lin_b is not None and attr_str == 'fc':
            m.fc.bias = nn.Parameter(lin_b)
  for n, ch in m.named_children():
    place_quant(ch, lin_w, lin_b, c_path + '_' + n,)


def balanced_scale_calibration_fwd(memory_tuple, train_set_copy, known_cl,
                                   total_cl, model, device, data_manager,
                                   replay = True):
  
  if quantFWDWgt != 'mem':
    return
  
  global quantUpdateScalePhase
  quantUpdateScalePhase = True

  with torch.no_grad():
    mem_samples, mem_targets = memory_tuple
    samples_per_cl = mem_samples.shape[0] / len(np.unique(mem_targets))
    if replay:
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

      new_samples = np.concatenate([np.concatenate(new_samples[key], axis=0) for key in new_samples.keys()], axis=0)

      try:
        mem_samples = np.concatenate([mem_samples, new_samples], axis=0)
      except:
        mem_samples = np.reshape(mem_samples, (mem_samples.shape[0], new_samples.shape[1],))
        mem_samples = np.concatenate([mem_samples, new_samples], axis=0)

      del train_set_copy, train_loader_copy

    update_loader = DataLoader(
        DummyDataset(torch.tensor(mem_samples), torch.tensor(mem_targets), transforms.Compose([*data_manager._train_trsf, *data_manager._common_trsf]), datatype = 'HAR' if len(mem_samples.shape) <= 2 else 'image'),
        batch_size=len(mem_samples), shuffle=True
        )

    global quant_bin_use_perc
    quant_bin_use_perc = []
    for _, inputs, targets in update_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      model(inputs)
      break
    print("usage perc: ", torch.mean(torch.tensor(quant_bin_use_perc)))
    global quant_bin_use_hist
    quant_bin_use_hist.append(torch.mean(torch.tensor(quant_bin_use_perc)))
    global scale_library_hist
    global scale_library
    scale_library_hist.append(scale_library)
    del update_loader
  
  # update_scale(model)
  # either scale or tiles and no recursion
  
  # update_bits((mem_samples, mem_targets), model, device,
  #             data_manager, replay)
  # update_tiles((mem_samples, mem_targets), model, device,
  #             data_manager, replay)
  
  quantUpdateScalePhase = False


def update_bits(memory_tuple, model, device, data_manager, replay):
  updated = False
  global quant_bin_use_perc
  global quantBits
  
  print("usage perc: ", torch.mean(torch.tensor(quant_bin_use_perc)))
  # if torch.mean(torch.tensor(quant_bin_use_perc)) > quantUpdateHighThr:
  if torch.mean(torch.tensor(quant_bin_use_perc)) >= 0.9:
  
    quantBits = max(1, quantBits + 1)
    updated = True
    print('increased quantBits to ' + str(quantBits))
  # elif torch.mean(torch.tensor(quant_bin_use_perc)) < quantUpdateLowThr:
  elif torch.mean(torch.tensor(quant_bin_use_perc)) < 0.7:
    quantBits = min(32, quantBits - 1)
    updated = True
    print('decreased quantBits to ' + str(quantBits))
  
  if updated:
    balanced_scale_calibration_fwd(memory_tuple, None, None, None, model,
                                   device, data_manager, replay,
                                   )


def update_tiles(memory_tuple, model, device, data_manager, replay):
  updated = False
  global quantBlockSize
  # if torch.mean(torch.tensor(quant_bin_use_perc)) > quantUpdateHighThr:
  # import pdb; pdb.set_trace()
  print("usage perc: ", torch.mean(torch.tensor(quant_bin_use_perc)), quantBlockSize)
  # if torch.mean(torch.tensor(quant_bin_use_perc)) >= 1.0:
  if torch.mean(torch.tensor(quant_bin_use_perc)) >= 0.9:
  
    quantBlockSize = min(quantBlockSize * 2, 128)
    updated = True
    print('increased quantBlockSize to ' + str(quantBlockSize))
  # elif torch.mean(torch.tensor(quant_bin_use_perc)) < quantUpdateLowThr:
  # elif torch.mean(torch.tensor(quant_bin_use_perc)) < 0.5:
  elif torch.mean(torch.tensor(quant_bin_use_perc)) < 0.6:
    quantBlockSize = max(quantBlockSize // 2, 2)
    updated = True
    print('decreased quantBlockSize to ' + str(quantBlockSize))
  
  # if updated:
  #   balanced_scale_calibration_fwd(memory_tuple, None, None, None, model,
  #                                  device, data_manager, replay,
  #                                  )


def update_scale(m, c_path='',):
  # bit shift
  for attr_str in dir(m):
    if attr_str[:1] != '_':
      target_attr = getattr(m, attr_str)
      if isinstance(target_attr, Conv2d_Ours):
        if hasattr(target_attr, 'c1'):
          c_name = c_path + '_' + attr_str
          if scale_library[c_path + '_' + attr_str][1] > quantUpdateHighThr:
            with torch.no_grad():
              target_attr.weight /= 2
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              target_attr.sw *= 2
              # print('increased scale '+c_path + '_' + attr_str)
            # elif quant_no_update_perc[c_name.replace('_', '.')[1:] + '.weight'] > quantUpdateLowThr:
          elif scale_library[c_path + '_' + attr_str][1] < quantUpdateLowThr:
            with torch.no_grad():
              target_attr.weight *= 2
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              target_attr.sw /= 2
          
      if isinstance(target_attr, Linear_Ours): # or isinstance(target_attr,
        if hasattr(target_attr, 'c1'):
          c_name = c_path + '_' + attr_str
          if scale_library[c_path + '_' + attr_str][1] > quantUpdateHighThr:
            with torch.no_grad():
              target_attr.weight /= 2
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              target_attr.sw *= 2
          elif  scale_library[c_path + '_' + attr_str][1] < quantUpdateLowThr:
            with torch.no_grad():
              target_attr.weight *= 2
              target_attr.weight.data = torch.floor(target_attr.weight.data)
              target_attr.sw /= 2

          
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

  mx = torch.min(x.abs().max(), torch.ones_like(x[0]) * 1e+8) * scale  # torch.quantile(x.abs(), .99) # TODO optimize calibration
  scale = mx / (2**(quantBits - 1) - 1)
  x = torch.clamp(x, -mx, mx) / (scale + EPSILON)

  sign = torch.sign(x)
  xq_ord = torch.floor(x.abs())

  qe = x.abs() - xq_ord
  flip = torch.bernoulli(qe)
  xq = xq_ord + flip

  xq = torch.reshape(sign * xq, dim)
  return xq, scale


def dynamic_intQ(x, scale=None):
  # can only be used in BWD - not differentiable
  if scale is None:
    scale = global_args["quantile"]
  mx = x.abs().max() * scale
  scale = mx / (2**(quantBits - 1) - 1)
  x = torch.clamp(x, -mx, mx)
  # epsilion for vmap # TODO eps size?
  return torch.round(x / (scale + EPSILON)), scale

class dynamic_intQ_FWD(Function):

  @staticmethod
  def forward(ctx, x):
    mx = x.abs().max() * global_args["quantile"]
    ctx.mx = mx
    ctx.save_for_backward(x)
    if x.min() < 0:
      scale = mx / (2**(quantBits - 1) - 1)
      x = torch.clamp(x, -mx, mx)
    else:
      scale = mx / (2**(quantBits) - 1)
      x = torch.clamp(x, 0, mx)

    return torch.round(x / (scale + EPSILON) ), scale

  @staticmethod
  def backward(ctx, grad_output, grad_scale):
    # STE
    x, = ctx.saved_tensors
    local_mx = (2**(quantBits - 1) - 1)

    grad_output = torch.where(x > local_mx, torch.tensor(
        [0], dtype=x.dtype, device=x.device), grad_output)
    grad_output = torch.where(
        x < -local_mx, torch.tensor([0], dtype=x.dtype, device=x.device), grad_output)

    return grad_output, None


def dynamic_fpQ(x, exponent_bits, mantissa_bits):
  # Handle zero separately
  zero_mask = (x == 0)

  # Get absolute value and log2
  abs_data = torch.abs(x)
  # add epsilon to avoid log(0)
  log2_data = torch.log2(abs_data+EPSILON)

  # Compute exponent
  exponent = torch.floor(log2_data).int()

  # Apply bias to exponent
  bias = 2**(exponent_bits-1)-1
  biased_exponent = torch.clamp(exponent + bias, 0, 2**exponent_bits-1)

  # Compute mantissa
  try:
    mantissa = abs_data / torch.ldexp(torch.tensor(1.0), exponent) - 1
    mantissa = torch.clamp(mantissa, 0, 1-2**(-mantissa_bits))  # Ensure mantissa is in [0, 1-2**(-mantissa_bits)]
    mantissa = torch.round(mantissa * 2**mantissa_bits) / 2**mantissa_bits  # mantissa_bits mantissa
  except:
    import pdb; pdb.set_trace()
  # Combine components
  result = torch.ldexp(1.0 + mantissa, biased_exponent - bias)

  # Restore sign and zero
  result = torch.copysign(result, x)
  result[zero_mask] = 0

  return result

class dynamic_fpQ_FWD(Function):

  @staticmethod
  def forward(ctx, x):
    mx = x.abs().max() * global_args["quantile"]
    ctx.mx = mx
    ctx.save_for_backward(x)

    abs_x = x.abs()

    # Perform floating point quantization
    log2_data = torch.log2(abs_x + EPSILON)  # Avoid log(0)
    exponent = torch.floor(log2_data).int()
    bias = 2**(exponent_bits - 1) - 1
    biased_exponent = torch.clamp(exponent + bias, 0, 2**exponent_bits - 1)
    
    mantissa = abs_x / torch.ldexp(torch.tensor(1.0), exponent) - 1
    mantissa = torch.clamp(mantissa, 0, 1 - 2**(-mantissa_bits))
    mantissa = torch.round(mantissa * 2**mantissa_bits) / 2**mantissa_bits
    
    quantized = torch.ldexp(1.0 + mantissa, biased_exponent - bias)
    
    
    quantized = torch.where(abs_x == 0, 0, quantized)
    quantized = torch.copysign(quantized, x)

    return quantized

  @staticmethod
  def backward(ctx, grad_output):
    # STE
    x, = ctx.saved_tensors
    mx = ctx.mx

    # Calculate gradients for the forward quantization step
    grad_output = torch.where(x > mx, torch.tensor(
        [0], dtype=x.dtype, device=x.device), grad_output)
    grad_output = torch.where(
        x < -mx, torch.tensor([0], dtype=x.dtype, device=x.device), grad_output)

    return grad_output


class FLinearQ(torch.autograd.Function):
  generate_vmap_rule = True

  @staticmethod
  def forward(x, w, h_out, h_bs, sx, sw):

    # print(torch.unique(w).shape)
    if quantFWDWgt == 'mem':
      # print("------------------")
      # print(torch.unique(w).shape)
      mx = 2**(quantWgtStoreBits - 1)
      scale = mx / (2**(quantBits - 1) -1)
      w = torch.clamp(w, -mx-1, mx-1)
      # print(torch.unique(w).shape)
      w = torch.round(w / (scale + EPSILON))
      # print(torch.unique(w).shape)
      if quantUpdateScalePhase:
        # get usage percent of quant bins
        global quant_bin_use_perc
        # import pdb; pdb.set_trace()
        print(torch.unique(w).shape)
        quant_bin_use_perc.append(torch.unique(w).shape[0] / (2**(quantBits - 1) * 2))
      


    global current_uname
    if quantBlockSize < x.shape[1]:
      fin_output = 0 * F.linear(x[:,0:quantBlockSize], w[:,0:quantBlockSize])

      for i in range(int(np.ceil( x.shape[1]/quantBlockSize ))):
        try:
          output = F.linear(x[:,i*quantBlockSize:(i+1)*quantBlockSize], w[:,i*quantBlockSize:(i+1)*quantBlockSize])
        except:
          import pdb; pdb.set_trace()
        # requantize to acc BW (clamp to big values - no scale)
        if quantAccBWD == "int" and quantAccBits < 16:
          # already integers so only clamp required
          n = 2**quantAccBits / 2 - 1
          output = torch.clamp(output, -n, n)
        elif quantAccBWD == "fp":
          output = dynamic_fpQ(output, exponent_bits_acc, mantissa_bits_acc)

        if quantUpdateScalePhase:
          global scale_library
          global current_uname
          scale_library[current_uname] = (int(torch.sum(output == 0.))/np.prod(output.shape),
                                          max(int(torch.sum(output == n))/np.prod(output.shape),
                                              int(torch.sum(output == -n))/np.prod(output.shape)))
          
        fin_output += output
    else:
      fin_output = F.linear(x, w)
      # requantize to acc BW (clamp to big values - no scale)
      if quantAccBWD == "int" and quantAccBits < 16:
        n = 2**quantAccBits / 2 - 1
        fin_output = torch.clamp(fin_output, -n, n)
      elif quantAccBWD == "fp":
        import pdb; pdb.set_trace()
        fin_output = dynamic_fpQ(fin_output, exponent_bits_acc, mantissa_bits_acc)
  
    return fin_output * sw * sx


  @staticmethod
  def setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1],
                          inputs[2], inputs[3], inputs[4], inputs[5])

  @staticmethod
  def backward(ctx, grad_output):
    
    x, w, h_out, h_bs, sx, sw = ctx.saved_tensors

    if quantHadOff:
      w_h1 = w
    else:
      w_h1 = h_out @ w
    # requantize weights
    if quantRequantize:
      if quantBWDWgt == 'int':
        w_h1, swh1 = dynamic_intQ(w_h1, scale=1.)
      elif quantBWDWgt == 'fp':
        w_h1 = dynamic_fpQ(w_h1, exponent_bits, mantissa_bits)
        swh1 = torch.tensor([1.0])
        swh1 = swh1.to(w.device)
      elif quantBWDWgt == 'noq':
        w_h1 = w_h1
        swh1 = torch.tensor([1.0])
        swh1 = swh1.to(w.device)
      else:
        raise Exception('Grad rounding scheme not implemented: ' + quantBWDWgt)
    else:
      swh1 = torch.tensor([1.0])
      swh1 = swh1.to(w.device)

    if quantHadOff:
      grad_output_h1 = grad_output
    else:  
      grad_output_h1 = grad_output @ h_out
    #get this vs grad_output
    # quant grad_output
    if quantBWDGrad1 == 'int':
      grad_output_h1, sg1 = dynamic_intQ(grad_output_h1)
    elif quantBWDGrad1 == 'fp':
      grad_output_h1 = dynamic_fpQ(grad_output_h1, exponent_bits, mantissa_bits)
      sg1 = torch.tensor([1.0])
      sg1 = sg1.to(w.device)
    elif quantBWDGrad1 == 'sq':
      raise Exception('not implemented')
      grad_output_h1 = dynamic_squant(grad_output_h1)
    elif quantBWDGrad1 == 'stoch':
      try:
        grad_output_h1, sg1 = dynamic_stoch(grad_output_h1)
      except:
        import pdb; pdb.set_trace()
    elif quantBWDGrad1 == 'noq':
      grad_output_h1 = grad_output_h1
      sg1 = torch.tensor([1.0])
      sg1 = sg1.to(w.device)
    else:
      raise Exception('Grad rounding scheme not implemented: ' + quantBWDGrad1)

    grad_input = (grad_output_h1 @ w_h1) 
    grad_input_noq = grad_output @ w
    
    global global_args
    if global_args["rec_grads"]:
      grad_input_copy = grad_input.clone().detach()
   
    if quantAccBWD == "int" and quantAccBits < 16:
      n = 2**quantAccBits / 2 - 1
      grad_input = torch.clamp(grad_input, -n, n)
    elif quantAccBWD == "fp":
      grad_input = dynamic_fpQ(grad_input, exponent_bits_acc, mantissa_bits_acc)
    # get bias 
    # or kl div
    global batchnr
    global epochnr
    if batchnr==0 and epochnr%50==0 and global_args["rec_grads"]:
      global gradient_library
      global current_uname
      if grad_output_h1.shape[1] < grad_output_h1.shape[0]:
        uname = "fc"
      elif grad_output_h1.shape[1] > grad_output_h1.shape[0]:
        try:
          if len(gradient_library["hidden1"]["gradnoq"]) > len(gradient_library["hidden2"]["gradnoq"]):
            uname = "hidden2"
          else:
            uname = "hidden1"
        except:
          if "hidden1" not in gradient_library:
            uname = "hidden1"
          else:
            uname = "hidden2"
      else:
        raise Exception("check weird")
      if uname not in gradient_library:
        gradient_library[uname] = {"gradnoq": [], "gradq": [], "gradqacc": []}
      gradient_library[uname]["gradnoq"].append(grad_input_noq.detach().cpu().numpy())
      gradient_library[uname]["gradq"].append(grad_input_copy.cpu().numpy())
      gradient_library[uname]["gradqacc"].append(grad_input.detach().cpu().numpy())
      # import pdb; pdb.set_trace()



    if quantHadOff:
      x_h2 = x
    else:
      x_h2 = h_bs @ x
    # requantize acts
    if quantRequantize:
      if quantBWDAct == 'int':
        x_h2, sxh2 = dynamic_intQ(x_h2, scale=1.)
      elif quantBWDAct == 'fp':
        x_h2 = dynamic_fpQ(x_h2, exponent_bits, mantissa_bits)
        sxh2 = torch.tensor([1.0])
        sxh2 = sxh2.to(w.device)
      elif quantBWDAct == 'noq':
        x_h2 = x_h2
        sxh2 = torch.tensor([1.0])
        sxh2 = sxh2.to(w.device)
      elif quantBWDAct == 'stoch':
        x_h2, sxh2 = dynamic_stoch(x_h2)
      else:
        raise Exception('Grad rounding scheme not implemented: ' + quantBWDAct)
    else:
      sxh2 = torch.tensor([1.0])
      sxh2 = sxh2.to(w.device)

    if quantHadOff:
      grad_output_h2 = grad_output.T
    else:
      grad_output_h2 = grad_output.T @ h_bs
    # quant grad_output
    if quantBWDGrad2 == 'int':
      grad_output_h2, sg2 = dynamic_intQ(grad_output_h2)
    elif quantBWDGrad2 == 'fp':
      grad_output_h2 = dynamic_fpQ(grad_output_h2, exponent_bits, mantissa_bits)
      sg2 = torch.tensor([1.0])
      sg2 = sg2.to(w.device)
    elif quantBWDGrad2 == 'sq':
      raise Exception('not implemented')
      grad_output_h2 = dynamic_squant(grad_output_h2)
    elif quantBWDGrad2 == 'stoch':
      grad_output_h2, sg2 = dynamic_stoch(grad_output_h2)
    elif quantBWDGrad2 == 'noq':
      grad_output_h2 = grad_output_h2
      sg2 = torch.tensor([1.0])
      sg2 = sg2.to(w.device)
    else:
      raise Exception('Grad rounding scheme not implemented: ' + quantBWDGrad2)

    grad_w = (grad_output_h2 @ x_h2) 

    if quantAccBWD == "int" and quantAccBits < 16:
      n = 2**quantAccBits / 2 - 1
      grad_w = torch.clamp(grad_w, -n, n)
    elif quantAccBWD == "fp":
      grad_w = dynamic_fpQ(grad_w, exponent_bits_acc, mantissa_bits_acc)

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


    if quantFWDWgt == 'mem':
      scale_dyn_range = global_args["dyn_scale"]
      with torch.no_grad():

        mx = self.weight.abs().max() * scale_dyn_range
        scale = mx / (2**(quantWgtStoreBits - 1) - 1)

        mult_mx = 2**(quantWgtStoreBits - 1) - 1
        mult_scale = mult_mx / (2**(quantBits - 1) - 1)

        self.register_buffer('sw', scale * mult_scale)
        self.weight.data = torch.round(self.weight/(scale + EPSILON))

  def forward(self, input):

    if quantFWDWgt == 'sawb':
      raise Exception('not implemented')
      w_q = UniformQuantizeSawb.apply(
          self.weight, self.c1, self.c2, self.QpW, self.QnW)
    elif quantFWDWgt == 'int':
      w_q, sw = dynamic_intQ_FWD.apply(self.weight)
    elif quantFWDWgt == 'fp':
      # import pdb; pdb.set_trace()
      w_q = dynamic_fpQ_FWD.apply(self.weight)
      sw = torch.tensor([1.0])
      sw = sw.to(self.weight.device)
    elif quantFWDWgt == 'lsq':
      raise Exception('not implemented')
      w_q = lsq(self.weight, self.lsq_wgt, self.QpW, self.QnW)
    elif quantFWDWgt == 'mem':
      w_q = self.weight
      sw = self.sw
    elif quantFWDWgt == 'noq':
      w_q = self.weight
      sw = torch.tensor([1.0])
      sw = sw.to(self.weight.device)
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
    elif quantFWDAct == 'fp':
      qinput = dynamic_fpQ_FWD.apply(input)
      sa = torch.tensor([1.0])
      sa = sa.to(self.weight.device)
    elif quantFWDAct == 'lsq':
      raise Exception('not implemented')
      qinput = lsq(input, self.lsq_wgt, self.QpA, self.QnA)
    elif quantFWDAct == 'noq':
      qinput = input
      sa = torch.tensor([1.0])
      sa = sa.to(self.weight.device)
    else:
      raise Exception(
          'FWD act quantized method not implemented: ' + quantFWDAct)

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

    if quantFWDWgt == 'mem':
      scale_dyn_range = global_args["dyn_scale"]
      with torch.no_grad():

        mx = self.weight.abs().max() * scale_dyn_range
        scale = mx / (2**(quantWgtStoreBits - 1) - 1)

        mult_mx = 2**(quantWgtStoreBits - 1) - 1
        mult_scale = mult_mx / (2**(quantBits - 1) - 1)

        self.register_buffer('sw', scale * mult_scale)
        self.weight.data = torch.round(self.weight/(scale + EPSILON))

  def forward(self, input):
    
    if quantFWDWgt == 'sawb':
      raise Exception('not implemented')
      w_q = UniformQuantizeSawb.apply(
          self.weight, self.c1, self.c2, self.QpW, self.QnW)
    elif quantFWDWgt == 'int':
      w_q, sw = dynamic_intQ_FWD.apply(self.weight)
    elif quantFWDWgt == 'fp':
      w_q = dynamic_fpQ_FWD.apply(self.weight)
      sw = torch.tensor([1.0])
    elif quantFWDWgt == 'lsq':
      raise Exception('not implemented')
      w_q = lsq(self.weight, self.lsq_wgt, self.QpW, self.QnW)
    elif quantFWDWgt == 'mem':
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
    elif quantFWDAct == 'fp':
      qinput = dynamic_fpQ_FWD.apply(input)
      sa = torch.tensor([1.0])
    elif quantFWDAct == 'lsq':
      raise Exception('not implemented')
      qinput = lsq(input, self.lsq_wgt, self.QpA, self.QnA)
    elif quantFWDAct == 'noq':
      qinput = input
      sa = torch.tensor([1.0])
    else:
      raise Exception(
          'FWD act quantized method not implemented: ' + quantFWDAct)

    # TODO: optimize speed of hadamard creation

    qinput = torch.nn.functional.unfold(
        qinput, self.kernel_size, padding=self.padding, stride=self.stride).transpose(1, 2)
    w_q = w_q.view(w_q.size(0), -1).t()

    if self.hadamard_bs.sum() == 0:
      self.hadamard_bs = torch.tensor(make_hadamard(
          qinput.shape[1]), dtype=self.weight.dtype).to(self.weight.device)

    flinearq_fn = torch.vmap(FLinearQ.apply, randomness='different')
    out = flinearq_fn(qinput, w_q.T.unsqueeze(0).repeat(qinput.shape[0], 1, 1), self.hadamard_out.unsqueeze(
        0).repeat(qinput.shape[0], 1, 1), self.hadamard_bs.unsqueeze(0).repeat(qinput.shape[0], 1, 1),
        sa.unsqueeze(0).repeat(qinput.shape[0], 1, 1), sw.unsqueeze(0).repeat(qinput.shape[0], 1, 1))

    # reshaping outputs into image form with batch, channel, height, width
    out = out.transpose(1, 2)
    output = out.view((input.shape[0], self.out_channels, int(
        input.shape[-2] / self.stride[0]), int(input.shape[-1] / self.stride[1])))

    if self.bias is not None:
      output += self.bias.view(1, -1, 1, 1)

    return output





class CosineLinear_Ours(CosineLinear):

  def __init__(self, uname, nb_proxy=1, to_reduce=False, sigma=True, *args,
               **kwargs):
    super(CosineLinear_Ours, self).__init__(nb_proxy=1, to_reduce=False,
                                            sigma=True,*args, **kwargs)
    init_properties(self, uname)

    self.register_buffer('hadamard_out', torch.tensor(
        make_hadamard(self.out_features), dtype=self.weight.dtype))
    self.register_buffer('hadamard_bs', torch.tensor(
        make_hadamard(quantBatchSize), dtype=self.weight.dtype))


    if quantFWDWgt == 'mem':
      scale_dyn_range = global_args["dyn_scale"]
      with torch.no_grad():

        mx = self.weight.abs().max() * scale_dyn_range
        scale = mx / (2**(quantWgtStoreBits - 1) - 1)

        mult_mx = 2**(quantWgtStoreBits - 1) - 1
        mult_scale = mult_mx / (2**(quantBits - 1) - 1)

        self.register_buffer('sw', scale * mult_scale)
        self.weight.data = torch.round(self.weight/(scale + EPSILON))

  def forward(self, input):
    self.weight = F.normalize(self.weight, p=2, dim=1)
    if quantFWDWgt == 'sawb':
      raise Exception('not implemented')
      w_q = UniformQuantizeSawb.apply(
          self.weight, self.c1, self.c2, self.QpW, self.QnW)
    elif quantFWDWgt == 'int':
      w_q, sw = dynamic_intQ_FWD.apply(self.weight)
    elif quantFWDWgt == 'fp':
      w_q = dynamic_fpQ_FWD.apply(self.weight)
      sw = torch.tensor([1.0])
    elif quantFWDWgt == 'lsq':
      raise Exception('not implemented')
      w_q = lsq(self.weight, self.lsq_wgt, self.QpW, self.QnW)
    elif quantFWDWgt == 'mem':
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

    input = F.normalize(input, p=2, dim=1)
    if quantFWDAct == 'sawb':
      raise Exception('not implemented')
      qinput = UniformQuantizeSawb.apply(
          input, self.c1, self.c2, self.QpA, self.QnA)
    elif quantFWDAct == 'int':
      qinput, sa = dynamic_intQ_FWD.apply(input)
    elif quantFWDAct == 'fp':
      qinput = dynamic_fpQ_FWD.apply(input)
      sa = torch.tensor([1.0])
    elif quantFWDAct == 'lsq':
      raise Exception('not implemented')
      qinput = lsq(input, self.lsq_wgt, self.QpA, self.QnA)
    elif quantFWDAct == 'noq':
      qinput = input
      sa = torch.tensor([1.0])
    else:
      raise Exception(
          'FWD act quantized method not implemented: ' + quantFWDAct)

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

    if self.to_reduce:
      # Reduce_proxy
      output = reduce_proxies(output, self.nb_proxy)

    if self.sigma is not None:
      output = self.sigma * output

    return {'logits': output}