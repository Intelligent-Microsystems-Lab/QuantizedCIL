import copy
import logging
import torch
from torch import nn
from backbones.cifar_resnet import resnet32
from backbones.resnet import resnet18, resnet34, resnet50
from backbones.ucir_cifar_resnet import resnet32 as cosine_resnet32
from backbones.ucir_resnet import resnet18 as cosine_resnet18
from backbones.ucir_resnet import resnet34 as cosine_resnet34
from backbones.ucir_resnet import resnet50 as cosine_resnet50
from backbones.linears import SimpleLinear, SplitCosineLinear, CosineLinear
from backbones.fc_net import FCNet, get_memo_fcnet

# FOR MEMO
from backbones.memo_resnet import get_resnet18_imagenet as get_memo_resnet18  # for imagenet
from backbones.memo_cifar_resnet import get_resnet32_a2fc as get_memo_resnet32  # for cifar

# FOR AUC & DER
from backbones.conv_cifar import conv2 as conv2_cifar
from backbones.cifar_resnet import resnet14 as resnet14_cifar
from backbones.cifar_resnet import resnet20 as resnet20_cifar
from backbones.cifar_resnet import resnet26 as resnet26_cifar

from backbones.conv_imagenet import conv4 as conv4_imagenet
from backbones.resnet import resnet10 as resnet10_imagenet
from backbones.resnet import resnet26 as resnet26_imagenet
from backbones.resnet import resnet34 as resnet34_imagenet
from backbones.resnet import resnet50 as resnet50_imagenet

# FOR AUC & MEMO
from backbones.conv_cifar import get_conv_a2fc as memo_conv2_cifar
from backbones.memo_cifar_resnet import get_resnet14_a2fc as memo_resnet14_cifar
from backbones.memo_cifar_resnet import get_resnet20_a2fc as memo_resnet20_cifar
from backbones.memo_cifar_resnet import get_resnet26_a2fc as memo_resnet26_cifar

from backbones.conv_imagenet import conv_a2fc_imagenet as memo_conv4_imagenet
from backbones.memo_resnet import get_resnet10_imagenet as memo_resnet10_imagenet
from backbones.memo_resnet import get_resnet26_imagenet as memo_resnet26_imagenet
from backbones.memo_resnet import get_resnet34_imagenet as memo_resnet34_imagenet
from backbones.memo_resnet import get_resnet50_imagenet as memo_resnet50_imagenet

import quant

def get_backbone(backbone_type, pretrained=False, args=None):
  name = backbone_type.lower()
  if name == "resnet32":
    return resnet32()
  elif name == "resnet18":
    return resnet18(pretrained=pretrained)
  elif name == "resnet34":
    return resnet34(pretrained=pretrained)
  elif name == "resnet50":
    return resnet50(pretrained=pretrained)
  elif name == "cosine_resnet18":
    return cosine_resnet18(pretrained=pretrained)
  elif name == "cosine_resnet32":
    return cosine_resnet32()
  elif name == "cosine_resnet34":
    return cosine_resnet34(pretrained=pretrained)
  elif name == "cosine_resnet50":
    return cosine_resnet50(pretrained=pretrained)
  elif name == 'cosine_fcnet':
    return FCNet(args["in_dim"], args["fc_hid_dim"], args["in_dim"],
                 args["fc_nr_hid"], "relu", args["bias"], args["half_dims"])
  # MEMO benchmark backbone
  elif name == 'memo_resnet18':
    _basenet, _adaptive_net = get_memo_resnet18()
    return _basenet, _adaptive_net
  elif name == 'memo_resnet32':
    _basenet, _adaptive_net = get_memo_resnet32()
    return _basenet, _adaptive_net
  elif name == 'memo_fcnet':
    _basenet, _adaptive_net = get_memo_fcnet(args["in_dim"], args["fc_hid_dim"],
                                             args["in_dim"],args["fc_nr_hid"],
                                             "relu", args["bias"],
                                             args["half_dims"])
    return _basenet, _adaptive_net
  # AUC
  # cifar
  elif name == 'conv2':
    return conv2_cifar()
  elif name == 'resnet14_cifar':
    return resnet14_cifar()
  elif name == 'resnet20_cifar':
    return resnet20_cifar()
  elif name == 'resnet26_cifar':
    return resnet26_cifar()

  elif name == 'memo_conv2':
    g_blocks, s_blocks = memo_conv2_cifar()  # generalized/specialized
    return g_blocks, s_blocks
  elif name == 'memo_resnet14_cifar':
    g_blocks, s_blocks = memo_resnet14_cifar()  # generalized/specialized
    return g_blocks, s_blocks
  elif name == 'memo_resnet20_cifar':
    g_blocks, s_blocks = memo_resnet20_cifar()  # generalized/specialized
    return g_blocks, s_blocks
  elif name == 'memo_resnet26_cifar':
    g_blocks, s_blocks = memo_resnet26_cifar()  # generalized/specialized
    return g_blocks, s_blocks

  # imagenet
  elif name == 'conv4':
    return conv4_imagenet()
  elif name == 'resnet10_imagenet':
    return resnet10_imagenet()
  elif name == 'resnet26_imagenet':
    return resnet26_imagenet()
  elif name == 'resnet34_imagenet':
    return resnet34_imagenet()
  elif name == 'resnet50_imagenet':
    return resnet50_imagenet()

  elif name == 'memo_conv4':
    g_blcoks, s_blocks = memo_conv4_imagenet()
    return g_blcoks, s_blocks
  elif name == 'memo_resnet10_imagenet':
    g_blcoks, s_blocks = memo_resnet10_imagenet()
    return g_blcoks, s_blocks
  elif name == 'memo_resnet26_imagenet':
    g_blcoks, s_blocks = memo_resnet26_imagenet()
    return g_blcoks, s_blocks
  elif name == 'memo_resnet34_imagenet':
    g_blocks, s_blocks = memo_resnet34_imagenet()
    return g_blocks, s_blocks
  elif name == 'memo_resnet50_imagenet':
    g_blcoks, s_blocks = memo_resnet50_imagenet()
    return g_blcoks, s_blocks
  elif name == 'fcnet' and args is not None:
    return FCNet(args["in_dim"], args["fc_hid_dim"], args["in_dim"],
                 args["fc_nr_hid"], "relu", args["bias"], args["half_dims"])

  else:
    raise NotImplementedError("Unknown type {}".format(backbone_type))


class BaseNet(nn.Module):
  def __init__(self, backbone_type, pretrained, args):
    super(BaseNet, self).__init__()
    self.backbone = get_backbone(backbone_type, pretrained, args)
    self.fc = None
    self.args = args

  @property
  def feature_dim(self):
    return self.backbone.out_dim

  def extract_vector(self, x):
    return self.backbone(x)["features"]

  def forward(self, x):
    x = self.backbone(x)
    out = self.fc(x["features"])
    """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
    out.update(x)

    return out

  def update_fc(self, nb_classes):
    pass

  def generate_fc(self, in_dim, out_dim):
    pass

  def copy(self):
    return copy.deepcopy(self)

  def freeze(self):
    for param in self.parameters():
      param.requires_grad = False
    self.eval()

    return self

  def load_checkpoint(self, args):
    if args["init_cls"] == 50:
      pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
          args["dataset"],
          args["seed"],
          args["backbone_type"],
          0,
          args["init_cls"],
      )
      checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
    else:
      checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
    model_infos = torch.load(checkpoint_name)
    self.backbone.load_state_dict(model_infos['backbone'])
    self.fc.load_state_dict(model_infos['fc'])
    test_acc = model_infos['test_acc']
    return test_acc


class IncrementalNet(BaseNet):
  def __init__(self, backbone_type, pretrained, gradcam=False, args=None):
    super().__init__(backbone_type, pretrained, args)
    self.gradcam = gradcam
    if hasattr(self, "gradcam") and self.gradcam:
      self._gradcam_hooks = [None, None]
      self.set_gradcam_hook()

  def update_fc(self, nb_classes):
    fc = self.generate_fc(self.feature_dim, nb_classes)
    if self.fc is not None:
      try:
        nb_output = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        fc.weight.data[:nb_output] = weight

        if self.fc.bias is not None:
          bias = copy.deepcopy(self.fc.bias.data)
          fc.bias.data[:nb_output] = bias
      except:
        nb_output = self.fc.module.out_features
        weight = copy.deepcopy(self.fc.module.weight.data)
        fc.weight.data[:nb_output] = weight

        if self.fc.module.bias is not None:
          bias = copy.deepcopy(self.fc.module.bias.data)
          fc.bias.data[:nb_output] = bias

    del self.fc
    self.fc = fc

  def weight_align(self, increment):
    weights = self.fc.weight.data
    newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
    oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
    meannew = torch.mean(newnorm)
    meanold = torch.mean(oldnorm)
    gamma = meanold / meannew
    print("alignweights,gamma=", gamma)
    self.fc.weight.data[-increment:, :] *= gamma

  def generate_fc(self, in_dim, out_dim):
    fc = SimpleLinear(in_dim, out_dim, bias=self.args["bias"])
    return fc

  def forward(self, x):
    x = self.backbone(x)
    out = self.fc(x["features"])
    out.update(x)
    if hasattr(self, "gradcam") and self.gradcam:
      out["gradcam_gradients"] = self._gradcam_gradients
      out["gradcam_activations"] = self._gradcam_activations

    return out

  def unset_gradcam_hook(self):
    self._gradcam_hooks[0].remove()
    self._gradcam_hooks[1].remove()
    self._gradcam_hooks[0] = None
    self._gradcam_hooks[1] = None
    self._gradcam_gradients, self._gradcam_activations = [None], [None]

  def set_gradcam_hook(self):
    self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def backward_hook(module, grad_input, grad_output):
      self._gradcam_gradients[0] = grad_output[0]
      return None

    def forward_hook(module, input, output):
      self._gradcam_activations[0] = output
      return None

    try:
      self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
          backward_hook
      )
      self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
          forward_hook
      )
    except:
      self._gradcam_hooks[0] = self.backbone.last_layer.register_backward_hook(
          backward_hook
      )
      self._gradcam_hooks[1] = self.backbone.last_layer.register_forward_hook(
          forward_hook
      )


class CosineIncrementalNet(BaseNet):
  def __init__(self, backbone_type, pretrained, nb_proxy=1):
    super().__init__(backbone_type, pretrained)
    self.nb_proxy = nb_proxy

  def update_fc(self, nb_classes, task_num):
    fc = self.generate_fc(self.feature_dim, nb_classes)
    if self.fc is not None:
      if task_num == 1:
        fc.fc1.weight.data = self.fc.weight.data
        fc.sigma.data = self.fc.sigma.data
      else:
        prev_out_features1 = self.fc.fc1.out_features
        fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
        fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
        fc.sigma.data = self.fc.sigma.data

    del self.fc
    self.fc = fc

  def generate_fc(self, in_dim, out_dim):
    if self.fc is None:
      fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
    else:
      prev_out_features = self.fc.out_features // self.nb_proxy
      # prev_out_features = self.fc.out_features
      fc = SplitCosineLinear(
          in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
      )

    return fc


class BiasLayer(nn.Module):
  def __init__(self):
    super(BiasLayer, self).__init__()
    self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
    self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

  def forward(self, x, low_range, high_range):
    ret_x = x.clone()
    ret_x[:, low_range:high_range] = (
        self.alpha * x[:, low_range:high_range] + self.beta
    )
    return ret_x

  def get_params(self):
    return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
  def __init__(self, backbone_type, pretrained, bias_correction=False, args=None):
    super().__init__(backbone_type, pretrained, args)

    # Bias layer
    self.bias_correction = bias_correction
    self.bias_layers = nn.ModuleList([])
    self.task_sizes = []

  def forward(self, x):
    x = self.backbone(x)
    out = self.fc(x["features"])
    if self.bias_correction:
      logits = out["logits"]
      for i, layer in enumerate(self.bias_layers):
        logits = layer(
            logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
        )
      out["logits"] = logits

    out.update(x)

    return out

  def update_fc(self, nb_classes):
    fc = self.generate_fc(self.feature_dim, nb_classes)
    if self.fc is not None:
      try:
        nb_output = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        bias = copy.deepcopy(self.fc.bias.data)
        fc.weight.data[:nb_output] = weight
        fc.bias.data[:nb_output] = bias
      except:
        nb_output = self.fc.module.out_features
        weight = copy.deepcopy(self.fc.module.weight.data)
        bias = copy.deepcopy(self.fc.module.bias.data)
        fc.weight.data[:nb_output] = weight
        fc.bias.data[:nb_output] = bias

    del self.fc
    self.fc = fc

    new_task_size = nb_classes - sum(self.task_sizes)
    self.task_sizes.append(new_task_size)
    self.bias_layers.append(BiasLayer())

  def generate_fc(self, in_dim, out_dim):
    fc = SimpleLinear(in_dim, out_dim)

    return fc

  def get_bias_params(self):
    params = []
    for layer in self.bias_layers:
      params.append(layer.get_params())

    return params

  def unfreeze(self):
    for param in self.parameters():
      param.requires_grad = True


class DERNet(nn.Module):
  def __init__(self, backbone_type, pretrained, args=None):
    super(DERNet, self).__init__()
    self.backbone_type = backbone_type
    self.backbones = nn.ModuleList()
    self.pretrained = pretrained
    self.out_dim = None
    self.fc = None
    self.aux_fc = None
    self.task_sizes = []
    self.args = args

  @property
  def feature_dim(self):
    if self.out_dim is None:
      return 0
    return self.out_dim * len(self.backbones)

  def extract_vector(self, x):
    features = [backbone(x)["features"] for backbone in self.backbones]
    features = torch.cat(features, 1)
    return features

  def forward(self, x):
    features = [backbone(x)["features"] for backbone in self.backbones]
    features = torch.cat(features, 1)

    out = self.fc(features)  # {logics: self.fc(features)}

    aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"]

    out.update({"aux_logits": aux_logits, "features": features})
    return out
    """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

  def update_fc(self, nb_classes):
    if len(self.backbones) == 0:
      self.backbones.append(get_backbone(self.backbone_type, args=self.args))
      lin_w, lin_b = quant.save_lin_params(self.backbones[-1])
      quant.place_quant(self.backbones[-1], lin_w, lin_b)
      # import pdb; pdb.set_trace()
    else:
      self.backbones.append(get_backbone(self.backbone_type, args=self.args))
      lin_w, lin_b = quant.save_lin_params(self.backbones[-1])
      quant.place_quant(self.backbones[-1], lin_w, lin_b)

      # test = [self.backbones[-1].state_dict()[key].shape for key in self.backbones[-1].state_dict()]
      # import pdb; pdb.set_trace()
      self.backbones[-1].load_state_dict({key: self.backbones[-2].state_dict()[key] for key in self.backbones[-2].state_dict() if (key in self.backbones[-1].state_dict() and  'hadamard' not in key)}, strict = False)


    if self.out_dim is None:
      self.out_dim = self.backbones[-1].out_dim
    fc = self.generate_fc(self.feature_dim, nb_classes)
    if self.fc is not None:
      try:
        nb_output = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
        if self.fc.bias is not None:
          bias = copy.deepcopy(self.fc.bias.data)
          fc.bias.data[:nb_output] = bias
      except:
        nb_output = self.fc.module.out_features
        weight = copy.deepcopy(self.fc.module.weight.data)
        self.fc.module.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
        # fc.weight.data[:nb_output] = weight

        if self.fc.module.bias is not None:
          bias = copy.deepcopy(self.fc.module.bias.data)
          fc.bias.data[:nb_output] = bias
    del self.fc
    self.fc = fc

    new_task_size = nb_classes - sum(self.task_sizes)
    self.task_sizes.append(new_task_size)

    self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

  def generate_fc(self, in_dim, out_dim):
    fc = SimpleLinear(in_dim, out_dim, bias=self.args["bias"])
    return fc

  def copy(self):
    return copy.deepcopy(self)

  def freeze(self):
    for param in self.parameters():
      param.requires_grad = False
    self.eval()

    return self

  def freeze_backbone(self):
    for param in self.backbones.parameters():
      param.requires_grad = False
    self.backbones.eval()

  def weight_align(self, increment):
    try:
      weights = self.fc.weight.data
    except:
      weights = self.fc.module.weight.data
    newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
    oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
    meannew = torch.mean(newnorm)
    meanold = torch.mean(oldnorm)
    gamma = meanold / meannew
    print("alignweights,gamma=", gamma)
    try:
      self.fc.weight.data[-increment:, :] *= gamma
    except:
      self.fc.module.weight.data[-increment:, :] *= gamma

  def load_checkpoint(self, args):
    checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
    model_infos = torch.load(checkpoint_name)
    assert len(self.backbones) == 1
    self.backbones[0].load_state_dict(model_infos['backbone'])
    self.fc.load_state_dict(model_infos['fc'])
    test_acc = model_infos['test_acc']
    return test_acc


class SimpleCosineIncrementalNet(BaseNet):
  def __init__(self, backbone_type, pretrained, args=None):
    super().__init__(backbone_type, pretrained, args)

  def update_fc(self, nb_classes, nextperiod_initialization):
    fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
    if self.fc is not None:
      try:
        nb_output = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        fc.sigma.data = self.fc.sigma.data
        if nextperiod_initialization is not None:
          weight = torch.cat([weight, nextperiod_initialization])
        fc.weight = nn.Parameter(weight)
      except:
        raise
        nb_output = self.fc.module.out_features
        weight = copy.deepcopy(self.fc.module.weight.data)
        fc.sigma.data = weight
    del self.fc
    self.fc = fc

  def generate_fc(self, in_dim, out_dim):
    fc = CosineLinear(in_dim, out_dim)
    return fc


class FOSTERNet(nn.Module):
  def __init__(self, backbone_type, pretrained, args=None):
    super(FOSTERNet, self).__init__()
    self.backbone_type = backbone_type
    self.backbones = nn.ModuleList()
    self.pretrained = pretrained
    self.out_dim = None
    self.fc = None
    self.fe_fc = None
    self.task_sizes = []
    self.oldfc = None
    self.args = args

  @property
  def feature_dim(self):
    if self.out_dim is None:
      return 0
    return self.out_dim * len(self.backbones)

  def extract_vector(self, x):
    features = [backbone(x)["features"] for backbone in self.backbones]
    features = torch.cat(features, 1)
    return features

  def forward(self, x):
    features = [backbone(x)["features"] for backbone in self.backbones]
    features = torch.cat(features, 1)
    out = self.fc(features)
    fe_logits = self.fe_fc(features[:, -self.out_dim:])["logits"]

    out.update({"fe_logits": fe_logits, "features": features})

    if self.oldfc is not None:
      old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
      out.update({"old_logits": old_logits})

    out.update({"eval_logits": out["logits"]})
    return out

  def update_fc(self, nb_classes):
    self.backbones.append(get_backbone(self.backbone_type, args=self.args))
    if self.out_dim is None:
      self.out_dim = self.backbones[-1].out_dim
    fc = self.generate_fc(self.feature_dim, nb_classes)
    if self.fc is not None:
      try:
        nb_output = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
        self.backbones[-1].load_state_dict(
          {key: self.backbones[-2].state_dict()[key] for key in self.backbones[-2].state_dict() if key in self.backbones[-1].state_dict()})
        if self.fc.bias is not None:
          bias = copy.deepcopy(self.fc.bias.data)
          fc.bias.data[:nb_output] = bias
      except:
        nb_output = self.fc.module.out_features
        weight = copy.deepcopy(self.fc.module.weight.data)
        fc.weight.data[:nb_output] = weight
        if self.fc.module.bias is not None:
          bias = copy.deepcopy(self.fc.module.bias.data)
          fc.bias.data[:nb_output] = bias
    self.oldfc = self.fc
    self.fc = fc
    new_task_size = nb_classes - sum(self.task_sizes)
    self.task_sizes.append(new_task_size)
    self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

  def generate_fc(self, in_dim, out_dim):
    fc = SimpleLinear(in_dim, out_dim, bias=self.args["bias"])
    return fc

  def copy(self):
    return copy.deepcopy(self)

  def copy_fc(self, fc):
    weight = copy.deepcopy(fc.weight.data)
    n, m = weight.shape[0], weight.shape[1]
    self.fc.weight.data[:n, :m] = weight
    try:
      bias = copy.deepcopy(fc.bias.data)
      self.fc.bias.data[:n] = bias
    except:
      pass

  def freeze(self):
    for param in self.parameters():
      param.requires_grad = False
    self.eval()
    return self

  def freeze_backbone(self):
    for param in self.backbones.parameters():
      param.requires_grad = False
    self.backbones.eval()

  def weight_align(self, old, increment, value):
    weights = self.fc.weight.data
    newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
    oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
    meannew = torch.mean(newnorm)
    meanold = torch.mean(oldnorm)
    gamma = meanold / meannew * (value ** (old / increment))
    logging.info("align weights, gamma = {} ".format(gamma))
    self.fc.weight.data[-increment:, :] *= gamma

  def load_checkpoint(self, args):
    if args["init_cls"] == 50:
      pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
          args["dataset"],
          args["seed"],
          args["backbone_type"],
          0,
          args["init_cls"],
      )
      checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
    else:
      checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
    model_infos = torch.load(checkpoint_name)
    assert len(self.backbones) == 1
    self.backbones[0].load_state_dict(model_infos['backbone'])
    self.fc.load_state_dict(model_infos['fc'])
    test_acc = model_infos['test_acc']
    return test_acc


class AdaptiveNet(nn.Module):
  def __init__(self, backbone_type, pretrained, args=None):
    super(AdaptiveNet, self).__init__()
    self.backbone_type = backbone_type
    self.TaskAgnosticExtractor, _ = get_backbone(
        backbone_type, pretrained, args)  # Generalized blocks
    self.TaskAgnosticExtractor.train()
    self.AdaptiveExtractors = nn.ModuleList()  # Specialized Blocks
    self.pretrained = pretrained
    self.out_dim = None
    self.fc = None
    self.aux_fc = None
    self.task_sizes = []
    self.args = args

  @property
  def feature_dim(self):
    if self.out_dim is None:
      return 0
    return self.out_dim * len(self.AdaptiveExtractors)

  def extract_vector(self, x):
    if "fcnet" in self.backbone_type:
      base_feature_map = self.TaskAgnosticExtractor(x)["features"]
      features = [extractor(base_feature_map)["features"]
                  for extractor in self.AdaptiveExtractors]
    else:
      base_feature_map = self.TaskAgnosticExtractor(x)
      features = [extractor(base_feature_map)
                  for extractor in self.AdaptiveExtractors]
    features = torch.cat(features, 1)
    return features

  def forward(self, x):
    if "fcnet" in self.backbone_type:
      base_feature_map = self.TaskAgnosticExtractor(x)["features"]
      features = [extractor(base_feature_map)["features"]
                  for extractor in self.AdaptiveExtractors]
    else:
      base_feature_map = self.TaskAgnosticExtractor(x)
      features = [extractor(base_feature_map)
                  for extractor in self.AdaptiveExtractors]
    features = torch.cat(features, 1)
    out = self.fc(features)  # {logits: self.fc(features)}

    aux_logits = self.aux_fc(features[:, -self.out_dim:])["logits"] 

    out.update({"aux_logits": aux_logits, "features": features})
    out.update({"base_features": base_feature_map})
    return out

    '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

  def update_fc(self, nb_classes):
    _, _new_extractor = get_backbone(self.backbone_type, self.pretrained,
                                     self.args)
    lin_w, lin_b = quant.save_lin_params(_new_extractor)
    quant.place_quant(_new_extractor, lin_w, lin_b)
    if len(self.AdaptiveExtractors) == 0:
      self.AdaptiveExtractors.append(_new_extractor)
    else:
      self.AdaptiveExtractors.append(_new_extractor)
      self.AdaptiveExtractors[-1].load_state_dict(
        {key: self.AdaptiveExtractors[-2].state_dict()[key] for key in self.AdaptiveExtractors[-2].state_dict() if (key in self.AdaptiveExtractors[-1].state_dict() and 'hadamard' not in key)}, strict = False)
      # self.AdaptiveExtractors[-1].load_state_dict(
      #     self.AdaptiveExtractors[-2].state_dict())

    if self.out_dim is None:
      logging.info(self.AdaptiveExtractors[-1])
      try:
        self.out_dim = self.AdaptiveExtractors[-1].feature_dim        
      except:
        self.out_dim = self.AdaptiveExtractors[-1].out_dim
    fc = self.generate_fc(self.feature_dim, nb_classes) 
    # lin_w, lin_b = quant.save_lin_params(fc)
    fc = quant.place_quant(fc, None, None, is_fc_layer = True)
    # import pdb; pdb.set_trace()
    if self.fc is not None:
      try:
        nb_output = self.fc.out_features
        weight = copy.deepcopy(self.fc.weight.data)
        fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight

        if self.fc.bias is not None:
          bias = copy.deepcopy(self.fc.bias.data)
          fc.bias.data[:nb_output] = bias
      except:
        nb_output = self.fc.module.out_features
        weight = copy.deepcopy(self.fc.module.weight.data)
        fc.module.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight

        if self.fc.module.bias is not None:
          bias = copy.deepcopy(self.fc.module.bias.data)
          fc.module.bias.data[:nb_output] = bias

    del self.fc
    self.fc = fc

    new_task_size = nb_classes - sum(self.task_sizes)
    self.task_sizes.append(new_task_size)
    self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

  def generate_fc(self, in_dim, out_dim):
    fc = SimpleLinear(in_dim, out_dim, bias=self.args["bias"])
    return fc

  def copy(self):
    return copy.deepcopy(self)

  def weight_align(self, increment):
    try:
      weights = self.fc.weight.data
    except:
      weights = self.fc.module.weight.data
    newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
    oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
    meannew = torch.mean(newnorm)
    meanold = torch.mean(oldnorm)
    gamma = meanold / meannew
    print('alignweights,gamma=', gamma)
    try:
      self.fc.weight.data[-increment:, :] *= gamma
    except:
      self.fc.module.weight.data[-increment:, :] *= gamma
  def load_checkpoint(self, args):
    if args["init_cls"] == 50:
      pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
          args["dataset"],
          args["seed"],
          args["backbone_type"],
          0,
          args["init_cls"],
      )
      checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
    else:
      checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
    checkpoint_name = checkpoint_name.replace("memo_", "")
    model_infos = torch.load(checkpoint_name)
    model_dict = model_infos['backbone']
    assert len(self.AdaptiveExtractors) == 1

    base_state_dict = self.TaskAgnosticExtractor.state_dict()
    adap_state_dict = self.AdaptiveExtractors[0].state_dict()

    pretrained_base_dict = {
        k: v
        for k, v in model_dict.items()
        if k in base_state_dict
    }

    pretrained_adap_dict = {
        k: v
        for k, v in model_dict.items()
        if k in adap_state_dict
    }

    base_state_dict.update(pretrained_base_dict)
    adap_state_dict.update(pretrained_adap_dict)

    self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
    self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
    self.fc.load_state_dict(model_infos['fc'])
    test_acc = model_infos['test_acc']
    return test_acc