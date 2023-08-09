import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
import copy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

from datetime import datetime
import quant

init_epoch = 170
init_lr = 0.05
init_milestones = [60, 100, 140]
init_lr_decay = 0.1
init_weight_decay = 2e-4  # 0.0005


epochs = 170  # 250
lrate = 0.05
milestones = [60, 100, 140]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2
lamda = 3


track_layer_list = ['_convnet_conv_1_3x3', '_convnet_stage_1_2_conv_b',
                    '_convnet_stage_2_4_conv_a', '_convnet_stage_3_3_conv_a', '_fc']
grad_quant_bias = {}


class LwF(BaseLearner):
  def __init__(self, args):
    super().__init__(args)
    self._network = IncrementalNet(args["convnet_type"], False)
    self.date_str = datetime.now().strftime('%y_%m_%d_%H_%M')

  def after_task(self):
    self._old_network = self._network.copy().freeze()
    self._known_classes = self._total_classes

  def incremental_train(self, data_manager):
    self._cur_task += 1
    self._total_classes = self._known_classes + data_manager.get_task_size(
        self._cur_task
    )
    self._network.update_fc(self._total_classes)
    logging.info(
        "Learning on {}-{}".format(self._known_classes, self._total_classes)
    )

    lin_w, lin_b = quant.save_lin_params(self._network)
    if quant.quantTrack:
      quant.place_track(self._network, track_layer_list, '', lin_w, lin_b)
    elif quant.quantMethod is not None:
      quant.place_quant(self._network, lin_w, lin_b)
    else:
      pass

    train_dataset = data_manager.get_dataset(
        np.arange(self._known_classes, self._total_classes),
        source="train",
        mode="train",
    )
    self.train_loader = DataLoader(
        train_dataset, batch_size=self.args['batch_size'], shuffle=True,
        num_workers=self.args['num_workers']
    )
    test_dataset = data_manager.get_dataset(
        np.arange(0, self._total_classes), source="test", mode="test"
    )
    self.test_loader = DataLoader(
        test_dataset, batch_size=self.args['batch_size'], shuffle=False,
        num_workers=self.args['num_workers']
    )

    if len(self._multiple_gpus) > 1:
      self._network = nn.DataParallel(self._network, self._multiple_gpus)
    self._train(self.train_loader, self.test_loader)
    if len(self._multiple_gpus) > 1:
      self._network = self._network.module

  def _train(self, train_loader, test_loader):
    self._network.to(self._device)
    if self._old_network is not None:
      self._old_network.to(self._device)

    if self._cur_task == 0:
      optimizer = optim.SGD(
          self._network.parameters(),
          momentum=0.9,
          lr=self.args['init_lr'],
          weight_decay=self.args['init_weight_decay'],
      )
      scheduler = optim.lr_scheduler.MultiStepLR(
          optimizer=optimizer, milestones=self.args['init_milestones'],
          gamma=self.args['init_lr_decay']
      )
      if self.args['skip']:
        if len(self._multiple_gpus) > 1:
          self._network = self._network.module
        load_acc = self._network.load_checkpoint(self.args)
        self._network.to(self._device)
        cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
        logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        if len(self._multiple_gpus) > 1:
          self._network = nn.DataParallel(self._network, self._multiple_gpus)
      else:
        self._init_train(train_loader, test_loader, optimizer, scheduler)
    else:
      optimizer = optim.SGD(
          self._network.parameters(),
          lr=self.args['lr'],
          momentum=0.9,
          weight_decay=self.args['weight_decay'],
      )
      scheduler = optim.lr_scheduler.MultiStepLR(
          optimizer=optimizer, milestones=self.args['milestones'],
          gamma=self.args['lr_decay']
      )
      self._update_representation(
          train_loader, test_loader, optimizer, scheduler)

    if quant.quantTrack:
        # save grads
      for gen_stats in ['train_acc', 'test_acc', 'loss']:
        np.save('track_stats/' + self.date_str + '_' + self.args['dataset'] + '_' + self.args['model_name'] + '_' + str(
            self._cur_task) + '_' + gen_stats + '.npy', quant.track_stats[gen_stats])
      for lname in track_layer_list:
        if lname in quant.track_stats['grads']:
          np.save('track_stats/' + self.date_str + '_' + self.args['dataset'] + '_' + self.args['model_name'] + '_' + str(
              self._cur_task) + lname + '.npy', torch.hstack(quant.track_stats['grads'][lname]).numpy())
        if lname in quant.track_stats['grads']:
          for stat_name in ['max', 'min', 'mean', 'norm']:
            np.save('track_stats/' + self.date_str + '_' + self.args['dataset'] + '_' + self.args['model_name'] + '_' + str(
                self._cur_task) + lname + '_' + stat_name + '.npy', torch.hstack(quant.track_stats['grad_stats'][lname][stat_name]).numpy())

  def _init_train(self, train_loader, test_loader, optimizer, scheduler):
    prog_bar = tqdm(range(self.args['init_epoch']))
    for _, epoch in enumerate(prog_bar):
      self._network.train()
      losses = 0.0
      correct, total = 0, 0
      for i, (_, inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        # unquantized tracking
        # quant.calibrate_phase = True
        logits = self._network(inputs)["logits"]
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()

        # # save all gradients
        # unquantized_grad = {}
        # for k, param in self._network.named_parameters():
        #   if 'weight' in k:
        #     if param.grad is not None:
        #       unquantized_grad[k] = copy.deepcopy(param.grad)

        optimizer.step()
        losses += loss.item()

        # quantized tracking
        # quant.calibrate_phase = False
        # logits = self._network(inputs)["logits"]
        # loss = F.cross_entropy(logits, targets)
        # optimizer.zero_grad()
        # loss.backward()

        # for k, param in self._network.named_parameters():
        #   if 'weight' in k:
        #     if param.grad is not None:
        #       if k in grad_quant_bias:
        #         grad_quant_bias[k] = .9 * grad_quant_bias[k] +  .1 * torch.mean(param.grad - unquantized_grad[k])
        #       else:
        #         grad_quant_bias[k] = torch.mean(unquantized_grad[k] - param.grad)

        _, preds = torch.max(logits, dim=1)
        local_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
        correct += local_correct
        total += len(targets)

        local_train_acc = np.around(tensor2numpy(
            local_correct) * 100 / len(targets), decimals=2)

      train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

      if epoch % 5 == 0:
        test_acc = self._compute_accuracy(self._network, test_loader)
        self._network.train()

        if quant.quantTrack:
          quant.track_stats['train_acc'].append(local_train_acc)
          quant.track_stats['test_acc'].append(test_acc)
          quant.track_stats['loss'].append(float(loss))
        info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
            self._cur_task,
            epoch + 1,
            self.args['epochs'],
            losses / len(train_loader),
            train_acc,
            test_acc,
        )
      else:
        info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
            self._cur_task,
            epoch + 1,
            self.args['epochs'],
            losses / len(train_loader),
            train_acc,
        )
      scheduler.step()
      prog_bar.set_description(info)
    logging.info(info)

  def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

    prog_bar = tqdm(range(self.args['epochs']))
    for _, epoch in enumerate(prog_bar):
      self._network.train()
      losses = 0.0
      correct, total = 0, 0
      for i, (_, inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        logits = self._network(inputs)["logits"]

        fake_targets = targets - self._known_classes
        loss_clf = F.cross_entropy(
            logits[:, self._known_classes:], fake_targets
        )
        loss_kd = _KD_loss(
            logits[:, : self._known_classes],
            self._old_network(inputs)["logits"],
            self.args['T'],
        )

        loss = self.args['lamda'] * loss_kd + loss_clf

        optimizer.zero_grad()
        loss.backward()

        # for k, param in self._network.named_parameters():
        #   if 'weight' in k:
        #     if param.grad is not None:
        #       param.grad += grad_quant_bias[k]

        optimizer.step()
        losses += loss.item()

        _, preds = torch.max(logits, dim=1)
        local_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
        correct += local_correct
        total += len(targets)

        local_train_acc = np.around(tensor2numpy(
            local_correct) * 100 / len(targets), decimals=2)

      train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

      if epoch % 5 == 0:
        test_acc = self._compute_accuracy(self._network, test_loader)
        self._network.train()

        if quant.quantTrack:
          quant.track_stats['train_acc'].append(local_train_acc)
          quant.track_stats['test_acc'].append(test_acc)
          quant.track_stats['loss'].append(float(loss))
        info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
            self._cur_task,
            epoch + 1,
            self.args['epochs'],
            losses / len(train_loader),
            train_acc,
            test_acc,
        )
      else:
        info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
            self._cur_task,
            epoch + 1,
            self.args['epochs'],
            losses / len(train_loader),
            train_acc,
        )
      scheduler.step()
      prog_bar.set_description(info)
    logging.info(info)


def _KD_loss(pred, soft, T):
  pred = torch.log_softmax(pred / T, dim=1)
  soft = torch.softmax(soft / T, dim=1)
  return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
