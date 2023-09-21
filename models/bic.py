import logging
import numpy as np
import torch
from torch import nn
from torch import optim
import copy
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNetWithBias

from datetime import datetime
import quant

track_layer_list = ['_convnet_conv_1_3x3', '_convnet_stage_1_2_conv_b',
                    '_convnet_stage_2_4_conv_a', '_convnet_stage_3_3_conv_a', '_fc']
grad_quant_bias = {}


class BiC(BaseLearner):
  def __init__(self, args):
    super().__init__(args)
    self._network = IncrementalNetWithBias(
        args["model_type"], False, bias_correction=True, args=args
    )
    self._class_means = None
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

    if self._cur_task >= 1:
      train_dset, val_dset = data_manager.get_dataset_with_split(
          np.arange(self._known_classes, self._total_classes),
          source="train",
          mode="train",
          appendent=self._get_memory(),
          val_samples_per_class=int(
              self.args['split_ratio']
              * self._memory_size / self._known_classes
          ),
      )
      self.val_loader = DataLoader(
          val_dset, batch_size=self.args['batch_size'], shuffle=True,
          num_workers=self.args['num_workers']
      )
      logging.info(
          "Stage1 dset: {}, Stage2 dset: {}".format(
              len(train_dset), len(val_dset)
          )
      )
      self.lamda = self._known_classes / self._total_classes
      logging.info("Lambda: {:.3f}".format(self.lamda))
    else:
      train_dset = data_manager.get_dataset(
          np.arange(self._known_classes, self._total_classes),
          source="train",
          mode="train",
          appendent=self._get_memory(),
      )
    test_dset = data_manager.get_dataset(
        np.arange(0, self._total_classes), source="test", mode="test"
    )

    self.train_loader = DataLoader(
        train_dset, batch_size=self.args['batch_size'], shuffle=True,
        num_workers=self.args['num_workers']
    )
    self.test_loader = DataLoader(
        test_dset, batch_size=self.args['batch_size'], shuffle=False,
        num_workers=self.args['num_workers']
    )
    # changed position to before training
    self.build_rehearsal_memory(data_manager, self.samples_per_class)

    self._log_bias_params()
    self._stage1_training(self.train_loader, self.test_loader, data_manager)
    
    if self._cur_task >= 1:
      self._stage2_bias_correction(self.val_loader, self.test_loader, data_manager)


    if len(self._multiple_gpus) > 1:
      self._network = self._network.module
    self._log_bias_params()

    # if quant.quantTrack:
    #   # save grads
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
    repl_stage = "training"
    self._run(qreplay_loader, test_loader, optimizer, scheduler, repl_stage,
                     data_manager, nr_epochs=2)
    quant.quantBits = old_qbits


  def _run(self, train_loader, test_loader, optimizer, scheduler, stage,
           data_manager, nr_epochs=False):
    if nr_epochs:
      prog_bar = tqdm(range(nr_epochs))
    else:
      prog_bar = tqdm(range(self.args['epochs']))
    gen_cnt = 0
    for _, epoch in enumerate(prog_bar):  # range(1, epochs + 1):
      self._network.train()
      losses = 0.0
      for i, (_, inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        logits = self._network(inputs)["logits"]

        if stage == "training":
          clf_loss = F.cross_entropy(logits, targets)
          if self._old_network is not None:
            old_logits = self._old_network(inputs)["logits"].detach()
            hat_pai_k = F.softmax(old_logits / self.args['T'], dim=1)
            log_pai_k = F.log_softmax(
                logits[:, : self._known_classes] / self.args['T'], dim=1
            )
            distill_loss = -torch.mean(
                torch.sum(hat_pai_k * log_pai_k, dim=1)
            )
            loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
          else:
            loss = clf_loss
        elif stage == "bias_correction":
          loss = F.cross_entropy(torch.softmax(logits, dim=1), targets)
        else:
          raise NotImplementedError()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()

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

      if epoch % 5 == 0:
        train_acc = self._compute_accuracy(self._network, train_loader)
        test_acc = self._compute_accuracy(self._network, test_loader)
        self._network.train()

        if quant.quantTrack:
          quant.track_stats["train_acc"].append(train_acc)
          quant.track_stats["test_acc"].append(test_acc)
          quant.track_stats["loss"].append(float(loss))
        info = "{} => Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}".format(
            stage,
            self._cur_task,
            epoch,
            self.args['epochs'],
            losses / len(train_loader),
            train_acc,
            test_acc,
        )
      else:
        info = "{} => Task {}, Epoch {}/{} => Loss {:.3f}".format(
            stage,
            self._cur_task,
            epoch,
            self.args['epochs'],
            losses / len(train_loader),
        )

      scheduler.step()
      prog_bar.set_description(info)
    logging.info(info)

  def _stage1_training(self, train_loader, test_loader, data_manager):
    """
    if self._cur_task == 0:
        loaded_dict = torch.load('./dict_0.pkl')
        self._network.load_state_dict(loaded_dict['model_state_dict'])
        self._network.to(self._device)
        return
    """

    ignored_params = list(map(id, self._network.bias_layers.parameters()))
    base_params = filter(
        lambda p: id(p) not in ignored_params, self._network.parameters()
    )
    network_params = [
        {"params": base_params,
         "lr": self.args['lr'], "weight_decay": self.args['weight_decay']},
        {
            "params": self._network.bias_layers.parameters(),
            "lr": 0,
            "weight_decay": 0,
        },
    ]
    if self.args["quantMethod"] == "fp134" or self.args["quantMethod"] == "fp130":
      optimizer = lp.optim.SGD(
          network_params,
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
          network_params,
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
          network_params,
          momentum=0.9,
          lr=self.args['init_lr'],
      )
      # never use 
      scheduler = optim.lr_scheduler.StepLR(
          optimizer=optimizer, step_size=1e32, gamma=1
      )
    else:
      raise NotImplementedError

    if len(self._multiple_gpus) > 1:
      self._network = nn.DataParallel(self._network, self._multiple_gpus)
    self._network.to(self._device)
    if self._old_network is not None:
      self._old_network.to(self._device)

    if self._cur_task == 0 and self.args['skip']:
      if len(self._multiple_gpus) > 1:
        self._network = self._network.module
      load_acc = self._network.load_checkpoint(self.args)
      self._network.to(self._device)
      cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
      logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
      if len(self._multiple_gpus) > 1:
        self._network = nn.DataParallel(self._network, self._multiple_gpus)
    else:
      self._run(train_loader, test_loader, optimizer,
                scheduler, stage="training", data_manager = data_manager)
    
    if self.args["quantReplaySize"]>0:
      mem_samples, mem_targets = self._get_memory()
      self.replay_train(data_manager, mem_samples, mem_targets)

  def _stage2_bias_correction(self, val_loader, test_loader, data_manager):
    if isinstance(self._network, nn.DataParallel):
      self._network = self._network.module
    network_params = [
        {
            "params": self._network.bias_layers[-1].parameters(),
            "lr": self.args['lr'],
            "weight_decay": self.args['weight_decay'],
        }
    ]
    if self.args["optimizer"] == "sgd":
      optimizer = optim.SGD(
          network_params,
          momentum=0.9,
          lr=self.args['lr'],
          weight_decay=self.args['weight_decay'],
      )
      scheduler = optim.lr_scheduler.MultiStepLR(
          optimizer=optimizer, milestones=self.args['milestones'],
          gamma=self.args['lr_decay']
      )
    elif self.args["optimizer"] == "ours":
      optimizer = quant.QuantMomentumOptimizer(
          network_params,
          momentum=0.9,
          lr=self.args['lr'],
      )
      # never use 
      scheduler = optim.lr_scheduler.StepLR(
          optimizer=optimizer, step_size=1e32, gamma=1
      )
    else:
      raise NotImplementedError

    if len(self._multiple_gpus) > 1:
      self._network = nn.DataParallel(self._network, self._multiple_gpus)
    self._network.to(self._device)

    self._run(
        val_loader, test_loader, optimizer, scheduler, stage="bias_correction", data_manager = data_manager
    )

  def _log_bias_params(self):
    logging.info("Parameters of bias layer:")
    params = self._network.get_bias_params()
    for i, param in enumerate(params):
      logging.info("{} => {:.3f}, {:.3f}".format(i, param[0], param[1]))
