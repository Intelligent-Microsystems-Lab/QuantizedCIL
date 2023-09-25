import torch
import torch.nn.utils.prune as prune
import numpy as np
from torch.utils.data import DataLoader
import copy
import pickle
import logging
from backbones.linears import SimpleLinear

import hashlib

def class_acc(model, data_manager, args):
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
  class_acc_train = {}
  class_acc_test = {}

  for i, (_, inputs, targets) in enumerate(test_loader):
    for j in targets:
      if j not in class_acc_train:
        class_acc_train[int(j.cpu())] = []

  for i, (_, inputs, targets) in enumerate(train_loader):
    for j in targets:
      if j not in class_acc_test:
        class_acc_test[int(j.cpu())] = []

  for i, (_, inputs, targets) in enumerate(test_loader):

    inputs, targets = inputs.to(model._device), targets.to(model._device)
    logits = model._network(inputs)["logits"]

    _, preds = torch.max(logits, dim=1)

    for j in range(len(targets)):
      class_acc_train[int(targets[j].cpu())].append( int((preds[j] == targets[j]).cpu()) )



  for i, (_, inputs, targets) in enumerate(train_loader):

    inputs, targets = inputs.to(model._device), targets.to(model._device)
    logits = model._network(inputs)["logits"]

    _, preds = torch.max(logits, dim=1)

    for j in range(len(targets)):
      class_acc_test[int(targets[j].cpu())].append( int((preds[j] == targets[j]).cpu()) )



  with open(f'dict_class_acc_{data_manager.dataset_name}_train.pkl', 'wb') as f:
    pickle.dump({x: np.mean(y) for x,y in class_acc_train.items()}, f)

  with open(f'dict_class_acc_{data_manager.dataset_name}_test.pkl', 'wb') as f:
    pickle.dump({x: np.mean(y) for x,y in class_acc_test.items()}, f)

  # import pdb; pdb.set_trace()
def class_acc_diff(model, data_manager, args):
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
  class_acc_train = {}
  class_acc_test = {}

  for i, (_, inputs, targets) in enumerate(test_loader):
    for j in targets:
      if j not in class_acc_train:
        class_acc_train[int(j.cpu())] = []

  for i, (_, inputs, targets) in enumerate(train_loader):
    for j in targets:
      if j not in class_acc_test:
        class_acc_test[int(j.cpu())] = []

  for i, (_, inputs, targets) in enumerate(test_loader):

    inputs, targets = inputs.to(model._device), targets.to(model._device)
    logits = model._network(inputs)["logits"]

    _, preds = torch.max(logits, dim=1)

    for j in range(len(targets)):
      class_acc_train[int(targets[j].cpu())].append( int((preds[j] == targets[j]).cpu()) )



  for i, (_, inputs, targets) in enumerate(train_loader):

    inputs, targets = inputs.to(model._device), targets.to(model._device)
    logits = model._network(inputs)["logits"]

    _, preds = torch.max(logits, dim=1)

    for j in range(len(targets)):
      class_acc_test[int(targets[j].cpu())].append( int((preds[j] == targets[j]).cpu()) )



  # with open(f'dict_class_acc_{data_manager.dataset_name}_train.pkl', 'wb') as f:
  #   train_og = pickle.load(f)

  # with open(f'dict_class_acc_{data_manager.dataset_name}_test.pkl', 'wb') as f:
  #   test_og = pickle.load(f)

  # 2023-09-24 12:51:43,577 [data_manager.py] => [19, 16, 2, 18, 7, 8, 15, 17, 10, 14, 6, 9, 5, 13, 11, 1, 12, 3, 4]

  # train_og = np.load(f'dict_class_acc_{data_manager.dataset_name}_train.pkl', allow_pickle=True

  # noCIL quant
  # {0: 0.0023809523809523725, 1: 0.0, 2: -0.03809523809523807, 3: 0.0, 4: -0.009523809523809601, 5: 0.06428571428571428, 6: -0.0071428571428571175, 7: 0.0, 8: -0.004761904761904745, 9: 0.0, 10: -0.004761904761904745, 11: -0.00952380952380949, 12: 0.004761904761904745, 13: 0.0, 14: 0.01904761904761909, 15: -0.32380952380952377, 16: 0.0, 17: -0.011904761904761973, 18: -0.02857142857142858}

  # just CIL
  # {0: -0.08571428571428574, 1: -0.011904761904761862, 2: -0.5738095238095238, 3: -0.08809523809523812, 4: -0.1523809523809524, 5: -0.07619047619047625, 6: -0.021428571428571463, 7: -0.02857142857142858, 8: -0.2833333333333333, 9: -0.030952380952380953, 10: -0.06190476190476191, 11: -0.033333333333333326, 12: -0.00952380952380949, 13: -0.014285714285714235, 14: -0.0071428571428571175, 15: -0.7023809523809523, 16: 0.0, 17: -0.9666666666666667, 18: 0.042857142857142816}

  # CIL quant
  # {0: -0.059523809523809534, 1: -0.02619047619047621, 2: -0.6857142857142857, 3: -0.04761904761904767, 4: -0.2523809523809524, 5: -0.03809523809523807, 6: -0.05238095238095242, 7: -0.05238095238095242, 8: -0.23333333333333328, 9: -0.004761904761904745, 10: -0.023809523809523836, 11: -0.09523809523809523, 12: -0.04047619047619044, 13: -0.011904761904761862, 14: -0.014285714285714235, 15: -0.85, 16: 0.0, 17: -0.8928571428571429, 18: 0.042857142857142816}

  test_og = np.load(f'dict_class_acc_{data_manager.dataset_name}_test.pkl', allow_pickle=True)

  diffs = {x: y-test_og[x] for x,y in {x: np.mean(y) for x,y in class_acc_test.items()}.items()}
  print(diffs)
  import pdb; pdb.set_trace()


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

  import pdb; pdb.set_trace()
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
