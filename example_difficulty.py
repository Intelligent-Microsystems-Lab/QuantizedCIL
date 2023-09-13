import torch
import torch.nn.utils.prune as prune
import numpy as np
from torch.utils.data import DataLoader
import copy
import pickle
import logging
from backbones.linears import SimpleLinear

import hashlib

def determine_difficulty(model, data_manager, args):

  test_dataset = data_manager.get_dataset(
        np.arange(0,  len(np.unique(data_manager._train_targets))), source="test", mode="test"
    )
  test_loader = DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['num_workers']
    )

  train_dataset = data_manager.get_dataset(
        np.arange(0,  len(np.unique(data_manager._train_targets))), source="train", mode="test"
    )

  train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['num_workers']
    )
  diff_dict = {}
  gen_cnt = 0
  for i, (_, inputs, targets) in enumerate(test_loader):
    for j in inputs:
      # if hashlib.sha512(j.cpu().numpy()).hexdigest() in diff_dict:
      #   print(hashlib.sha512(j.cpu().numpy()).hexdigest())
      diff_dict[hashlib.sha512(j.cpu().numpy()).hexdigest()] = -1
      gen_cnt += 1

  for i, (_, inputs, targets) in enumerate(train_loader):
    for j in inputs:
      # if hashlib.sha512(j.cpu().numpy()).hexdigest() in diff_dict:
      #   print(hashlib.sha512(j.cpu().numpy()).hexdigest())
      diff_dict[hashlib.sha512(j.cpu().numpy()).hexdigest()] = -1
      gen_cnt += 1

  for i in range(101):
    print(f'eval diff: {i}%')
    tmp_model = copy.deepcopy(model._network)

    
    parameters_to_prune = []
    for name, module in tmp_model.named_modules():
      if isinstance(module, torch.nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))
      elif isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))
      elif isinstance(module, SimpleLinear):
        parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(parameters_to_prune,pruning_method=prune.L1Unstructured,amount=i/100,)

    for i, (_, inputs, targets) in enumerate(test_loader):

      inputs, targets = inputs.to(model._device), targets.to(model._device)
      logits = tmp_model(inputs)["logits"]

      _, preds = torch.max(logits, dim=1)
      local_correct = preds.eq(targets.expand_as(preds)) # .cpu().sum()

      for idx, j in enumerate(local_correct):
        if j:
          diff_dict[hashlib.sha512(j.cpu().numpy()).hexdigest()] = i

    for i, (_, inputs, targets) in enumerate(train_loader):

      inputs, targets = inputs.to(model._device), targets.to(model._device)
      logits = tmp_model(inputs)["logits"]

      _, preds = torch.max(logits, dim=1)
      local_correct = preds.eq(targets.expand_as(preds)) # .cpu().sum()

      for idx, j in enumerate(local_correct):
        if j:
          diff_dict[hashlib.sha512(j.cpu().numpy()).hexdigest()] = i

    # import pdb; pdb.set_trace()
  with open(f'saved_dictionary_{data_manager.dataset_name}.pkl', 'wb') as f:
    pickle.dump(diff_dict, f)



def what_did_i_forget(model, data_manager, args):
  # compute average difficulty of examples which are not classified correctly
  # anymore.
  with open(f'saved_dictionary_{data_manager.dataset_name}.pkl', 'rb') as f:
    diff_dict = pickle.load(f)

  test_dataset = data_manager.get_dataset(
        np.arange(0,  len(np.unique(data_manager._train_targets))),
        source="test", mode="test"
    )
  test_loader = DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['num_workers']
    )

  score = 0
  cnt = 0
  for i, (_, inputs, targets) in enumerate(test_loader):

    inputs, targets = inputs.to(model._device), targets.to(model._device)
    logits = model._network(inputs)["logits"]

    _, preds = torch.max(logits, dim=1)
    local_correct = preds.eq(targets.expand_as(preds)) # .cpu().sum()

    for idx, j in enumerate(local_correct):
      if not j:
        uid = hashlib.sha512(inputs[idx].cpu().numpy()).hexdigest()
        if diff_dict[uid] != -1:
          score += diff_dict[uid]
          cnt += 1

  logging.info(f'Our quality lost score: {score/(cnt+1e32)}')
