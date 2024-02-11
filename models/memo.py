import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

try:
  import lptorch as lp
  qnn = lp.nn
except:
  # makes code executable without lptorch compiled
  pass

import quant

num_workers=8
T=2

class MEMO(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        self._network = AdaptiveNet(args['model_type'], False, args=args)
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
        
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        lin_w, lin_b = quant.save_lin_params(self._network)
        if quant.quantMethod is not None:
          if quant.quantMethod != 'noq':
            quant.place_quant(self._network, lin_w, lin_b)
        else:
          pass
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='train', 
            appendent=self._get_memory()
        )
        # import pdb; pdb.set_trace()
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=True, 
            num_workers=self.args['num_workers']
        )
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source='test', 
            mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"],
            shuffle=False, 
            num_workers=self.args['num_workers']
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()                   #All status from eval to train
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()
        
        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            if 'fp13' in self.args["quantMethod"]:
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
                    lr=self.args["init_lr"],
                    weight_decay=self.args["init_weight_decay"]
                )
                if self.args['scheduler'] == 'steplr':
                    scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer=optimizer, 
                        milestones=self.args['init_milestones'], 
                        gamma=self.args['init_lr_decay']
                    )
                elif self.args['scheduler'] == 'cosine':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=self.args['init_epoch']
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
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            if 'fp13' in self.args["quantMethod"]:
                optimizer = lp.optim.SGD(
                    self._network.parameters(),
                    momentum=0.9,
                    lr=self.args['lr'],
                    weight_decay=self.args['weight_decay'],
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
                    weight_decay=self.args['weight_decay']
                )
                if self.args['scheduler'] == 'steplr':
                    scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer=optimizer,
                        milestones=self.args['milestones'], 
                        gamma=self.args['lr_decay']
                    )
                elif self.args['scheduler'] == 'cosine':
                    # changed from t_max to epochs 
                    assert self.args['init_epoch'] is not None
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=self.args['init_epoch']
                    )
            elif self.args["optimizer"] == "ours":
                optimizer = quant.QuantMomentumOptimizer(
                    self._network.parameters(),
                    momentum=0.9,
                    lr=self.args['lr'],
                )
                # never use 
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer=optimizer, step_size=1e32, gamma=1
                )
            else:
                raise NotImplementedError
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
            else:
                self._network.weight_align(self._total_classes-self._known_classes)

            
    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)['logits']
                
                loss=F.cross_entropy(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            # prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.set_network()
            losses = 0.
            losses_clf=0.
            losses_aux=0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                outputs= self._network(inputs)
                logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                loss_clf=F.cross_entropy(logits,targets)
                aux_targets = targets.clone()
                aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
                loss_aux=F.cross_entropy(aux_logits,aux_targets)
                loss=loss_clf+self.args['alpha_aux']*loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux+=loss_aux.item()
                losses_clf+=loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux  {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
            prog_bar.set_description(info)
        logging.info(info)