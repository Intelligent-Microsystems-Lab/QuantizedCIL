import logging
import numpy as np
from tqdm import tqdm
import copy
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from torchvision import transforms

from utils.data_manager import DataManager
from utils.data_manager import DummyDataset

from datetime import datetime
import quant

import lptorch as lp

track_layer_list = ['_convnet_conv_1_3x3', '_convnet_stage_1_2_conv_b',
                    '_convnet_stage_2_4_conv_a', '_convnet_stage_3_3_conv_a', '_fc']
grad_quant_bias = {}

# bias_track = []
# var_track = []

# TODO @clee1994 turn off for speed
# torch.autograd.set_detect_anomaly(True)


class iCaRL(BaseLearner):
  def __init__(self, args):
    super().__init__(args)
    self._network = IncrementalNet(args["model_type"], False, args=args)
    self.date_str = datetime.now().strftime('%y_%m_%d_%H_%M')

  def after_task(self):
    self._old_network = self._network.copy().freeze()
    self._known_classes = self._total_classes
    logging.info("Exemplar size: {}".format(self.exemplar_size))

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
      if quant.quantMethod != 'noq':
        quant.place_quant(self._network, lin_w, lin_b)
    else:
      pass

    train_dataset = data_manager.get_dataset(
        np.arange(self._known_classes, self._total_classes),
        source="train",
        mode="train",
        appendent=self._get_memory(),
    )
    self.train_loader = DataLoader(
        train_dataset, batch_size=self.args['batch_size'], shuffle=True,
        num_workers=self.args['num_workers'],
    )
    test_dataset = data_manager.get_dataset(
        np.arange(0, self._total_classes), source="test", mode="test"
    )
    self.test_loader = DataLoader(
        test_dataset, batch_size=self.args['batch_size'], shuffle=False,
        num_workers=self.args['num_workers']
    )

    if self.args['skip'] and self._cur_task == 0:
      load_acc = self._network.load_checkpoint(self.args)

    if len(self._multiple_gpus) > 1:
      self._network = nn.DataParallel(self._network, self._multiple_gpus)

    if self._cur_task == 0:
      if self.args['skip']:
        self._network.to(self._device)
        cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
        logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
      else:
        self._train(self.train_loader, self.test_loader, data_manager) 
        self._compute_accuracy(self._network, self.test_loader)
    else:
      self._train(self.train_loader, self.test_loader, data_manager)

    self.build_rehearsal_memory(data_manager, self.samples_per_class)
    if len(self._multiple_gpus) > 1:
      self._network = self._network.module

  def _train(self, train_loader, test_loader, data_manager):
    self._network.to(self._device)
    if self._old_network is not None:
      self._old_network.to(self._device)

    if self._cur_task == 0:
      if self.args["quantMethod"] == "fp134" or self.args["quantMethod"] == "fp130":
        optimizer = lp.optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=self.args['init_lr'],
            weight_decay=self.args['init_weight_decay'],
            weight_quantize=False
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.args['init_milestones'],
            gamma=self.args['init_lr_decay']
        )
      elif self.args["optimizer"] == "sgd":
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
      elif self.args["optimizer"] == "ours":
        optimizer = quant.QuantMomentumOptimizer(
            self._network.parameters(),
            momentum=0.9,
            lr=self.args['init_lr'],
        )
        # never use 
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=1e32, gamma=1
        )
      else:
        raise NotImplementedError

      self._init_train(train_loader, test_loader, optimizer, scheduler, data_manager)
    else:
      if self.args["quantMethod"] == "fp134" or self.args["quantMethod"] == "fp130":
        optimizer = lp.optim.SGD(
            self._network.parameters(),
            momentum=0.9,
            lr=self.args['init_lr'],
            weight_decay=self.args['init_weight_decay'],
            weight_quantize=False
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.args['init_milestones'],
            gamma=self.args['init_lr_decay']
        )
      elif self.args["optimizer"] == "sgd":
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
      elif self.args["optimizer"] == "ours":
        optimizer = quant.QuantMomentumOptimizer(
            self._network.parameters(),
            momentum=0.9,
            lr=self.args['init_lr'],
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=self.args['init_milestones'],
            gamma=self.args['init_lr_decay']
        )
      else:
        raise NotImplementedError
      self._update_representation(
          train_loader, test_loader, optimizer, scheduler, data_manager)
      
    if self.args["quantReplaySize"] > 0 and self._cur_task > 0:
      mem_samples, mem_targets = self._get_memory()
      if len(mem_samples) > 0:
        self.replay_train(test_loader, optimizer, scheduler, data_manager, mem_samples, mem_targets)


    # with open('../bias_noquant.npy', 'wb') as f:
    #   np.save(f, np.array([float(x) for x in bias_track]))
    # with open('../var_noquant.npy', 'wb') as f:
    #   np.save(f, np.array([float(x) for x in var_track]))

    # if quant.quantTrack:
    #     # save grads
    #   for gen_stats in ['train_acc', 'test_acc', 'loss']:
    #     np.save('track_stats/' + self.date_str + '_' + self.args['dataset'] + '_' + self.args['model_name'] + '_' + str(
    #         self._cur_task) + '_' + gen_stats + '.npy', quant.track_stats[gen_stats])
    #   for lname in track_layer_list:
    #     if lname in quant.track_stats['grads']:
    #       np.save('track_stats/' + self.date_str + '_' + self.args['dataset'] + '_' + self.args['model_name'] + '_' + str(
    #           self._cur_task) + lname + '.npy', torch.hstack(quant.track_stats['grads'][lname]).numpy())
    #     if lname in quant.track_stats['grads']:
    #       for stat_name in ['max', 'min', 'mean', 'norm']:
    #         np.save('track_stats/' + self.date_str + '_' + self.args['dataset'] + '_' + self.args['model_name'] + '_' + str(
    #             self._cur_task) + lname + '_' + stat_name + '.npy', torch.hstack(quant.track_stats['grad_stats'][lname][stat_name]).numpy())
  def sample_n_p_c_from_memory(self, n, mem_samples, mem_targets):
    new_samples = {a:[] for a in np.unique(mem_targets)}
    for input, target in zip(mem_samples, mem_targets):
      if len(new_samples[target.item()]) < n:
        new_samples[target.item()].append(input)
        if sum([len(new_samples[key])==n for key in new_samples]) == len(list(new_samples.keys())):
          break
    new_targets = np.array([])
    for cl in np.unique(mem_targets):
      new_targets = np.concatenate([new_targets, torch.tensor([cl] * new_samples[cl].__len__() )])
    return new_samples, new_targets


  def replay_train(self, test_loader, optimizer, scheduler, data_manager, mem_samples, mem_targets):
    
    old_qbits = quant.quantBits
    old_accbits = quant.quantAccBits 
    quant.quantBits = 8
    quant.quantAccBits = quant.quantBits * 2
    
    samples_per_cl = int(self.args["quantReplaySize"] / len(np.unique(mem_targets)))
    if samples_per_cl == 0:
      samples_per_cl = 1
      print("Warning: higher bit replay size too small, using 1 sample per class")

    repl_smpls, repl_tgts = self.sample_n_p_c_from_memory(samples_per_cl, mem_samples, mem_targets)

    print("Higher precion replay:")
    qreplay_loader = DataLoader(
        DummyDataset(torch.tensor(mem_samples), torch.tensor(mem_targets),
                     transforms.Compose([*data_manager._train_trsf,]),
                     datatype = 'HAR' if len(mem_samples.shape) <= 2 else 'image'),
        batch_size=len(mem_samples), shuffle=True
        )
    self._init_train(qreplay_loader, test_loader, optimizer, scheduler,
                     data_manager, nr_epochs=2)
    quant.quantBits = old_qbits


  def _init_train(self, train_loader, test_loader, optimizer, scheduler, data_manager, nr_epochs=False):
    if nr_epochs:
      prog_bar = tqdm(range(nr_epochs))
    else:
      prog_bar = tqdm(range(self.args['init_epoch']))
    gen_cnt = 0
    for i, epoch in enumerate(prog_bar):

      self._network.train()
      losses = 0.0
      correct, total = 0, 0
      for i, (_, inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(self._device), targets.to(self._device)


        logits = self._network(inputs)["logits"]
        # import pdb; pdb.set_trace()
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        # import pdb; pdb.set_trace()
        # bias_track.append(self._network.backbone.net[1].lw.weight.grad.mean())
        # var_track.append(self._network.backbone.net[1].lw.weight.grad.std())

        # backup_w = copy.deepcopy({k:x for k, x in self._network.named_parameters()})
        optimizer.step()
        losses += loss.item()

        _, preds = torch.max(logits, dim=1)
        local_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
        correct += preds.eq(targets.expand_as(preds)).cpu().sum()
        total += len(targets)

        local_train_acc = np.around(tensor2numpy(
            local_correct) * 100 / len(targets), decimals=2)
        
        # set quant scale update
        if gen_cnt % self.args["quantUpdateP"] == 0 and gen_cnt > 0:
          with torch.no_grad():
            try:
              mem_samples, mem_targets = self._get_memory()
            except:
              mem_samples, mem_targets = np.zeros_like(inputs.cpu()), np.zeros_like(targets.cpu())
            # get as many samples from the new classes as for each in memory
            train_copy = data_manager.get_dataset(
                            np.arange(self._known_classes, self._total_classes),
                            source="train",
                            mode="train",
                            no_trsf = True,
                            )
            
            # with torch.no_grad():
            #   no_update_perc = { k: np.mean((backup_w[k] == v).cpu().numpy()) for k,v in self._network.named_parameters()}
            # quant.quant_no_update_perc = no_update_perc
            quant.balanced_scale_calibration_fwd((mem_samples, mem_targets), train_copy,
                                           self._known_classes, self._total_classes,
                                           self._network, inputs.device, data_manager)

        gen_cnt += 1
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

  def _update_representation(self, train_loader, test_loader, optimizer, scheduler, data_manager, nr_epochs=False):
    if nr_epochs:
      prog_bar = tqdm(range(nr_epochs))
    else:
      prog_bar = tqdm(range(self.args['epochs']))
    gen_cnt = 0
    for i, epoch in enumerate(prog_bar):

      self._network.train()
      losses = 0.0
      correct, total = 0, 0
      for i, (_, inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        logits = self._network(inputs)["logits"]
        # backup_w = copy.deepcopy({k:x for k, x in self._network.named_parameters()})

        loss_clf = F.cross_entropy(logits, targets)
        loss_kd = _KD_loss(
            logits[:, : self._known_classes],
            self._old_network(inputs)["logits"],
            self.args['T'],
        )

        loss = loss_clf + loss_kd
        optimizer.zero_grad()
        loss.backward()
        # import pdb; pdb.set_trace()
        # bias_track.append(self._network.backbone.net[1].lw.weight.grad.mean())
        # var_track.append(self._network.backbone.net[1].lw.weight.grad.std())

        optimizer.step()
        losses += loss.item()

        _, preds = torch.max(logits, dim=1)
        local_correct = preds.eq(targets.expand_as(preds)).cpu().sum()
        correct += preds.eq(targets.expand_as(preds)).cpu().sum()
        total += len(targets)

        local_train_acc = np.around(tensor2numpy(
            local_correct) * 100 / len(targets), decimals=2)
        
        # set quant scale update
        if gen_cnt % self.args["quantUpdateP"] == 0 and gen_cnt > 0:
          with torch.no_grad():
            mem_samples, mem_targets = self._get_memory()
            # get as many samples from the new classes as for each in memory
            train_copy = data_manager.get_dataset(
                            np.arange(self._known_classes, self._total_classes),
                            source="train",
                            mode="train",
                            no_trsf = True,
                            )
            # with torch.no_grad():
            #   no_update_perc = { k: np.mean((backup_w[k] == v).cpu().numpy()) for k,v in self._network.named_parameters()}
            # quant.quant_no_update_perc = no_update_perc
            quant.balanced_scale_calibration_fwd((mem_samples, mem_targets), train_copy,
                                           self._known_classes, self._total_classes,
                                           self._network, inputs.device, data_manager)
        
        gen_cnt += 1
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
