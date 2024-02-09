import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

try:
  import lptorch as lp
  qnn = lp.nn
except:
  # makes code executable without lptorch compiled
  pass

import quant 

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 170
lrate = 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 8
T = 2


class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = DERNet(args["model_type"], False, args=args)

    def after_task(self):
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
        if quant.quantMethod is not None:
          if quant.quantMethod != 'noq':
            quant.place_quant(self._network, lin_w, lin_b)
        else:
          pass
        
        #TODO 
        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.backbones[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        # import pdb; pdb.set_trace()
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
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.backbones[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.backbones[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
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
                filter(lambda p: p.requires_grad, self._network.parameters()),
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
          if not self.args['skip']:
              self._init_train(train_loader, test_loader, optimizer, scheduler)
          else:
              test_acc = self._network.load_checkpoint(self.args)
              cur_test_acc = self._compute_accuracy(self._network, test_loader)
              logging.info(f"Loaded Test Acc:{test_acc} Cur_Test_Acc:{cur_test_acc}")
                
        else:
          if self.args["quantMethod"] == "fp134" or self.args["quantMethod"] == "fp130":
            optimizer = lp.optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args['lr'],
                weight_decay=self.args['init_weight_decay'],
                weight_quantize=False
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['milestones'],
                gamma=self.args['lr_decay']
            )
          elif self.args["optimizer"] == "sgd":  
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args['lr'],
                momentum=0.9,
                weight_decay=self.args['weight_decay'],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['milestones'],
                gamma=self.args['lr_decay']
            )
          elif self.args["optimizer"] == "ours":
            optimizer = quant.QuantMomentumOptimizer(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args['lr'],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['milestones'],
                gamma=self.args['lr_decay']
            )
          else:
            raise NotImplementedError
          self._update_representation(train_loader, test_loader, optimizer,
                                      scheduler)
          if len(self._multiple_gpus) > 1:
              self._network.module.weight_align(
                  self._total_classes - self._known_classes
              )
          else:
              self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['init_epoch']))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['epochs']))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                loss = loss_clf + loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)