import torch
import torch.nn.utils.prune as prune
import numpy as np
from torch.utils.data import DataLoader
import copy
import pickle
import logging

def determine_difficulty(model, data_manager, args):

  test_dataset = data_manager.get_dataset(
        np.arange(0,  len(np.unique(data_manager._train_targets))), source="test", mode="test"
    )
  test_loader = DataLoader(
        test_dataset, batch_size=args['batch_size'], shuffle=False,
        num_workers=args['num_workers']
    )

  diff_dict = {}

  for i, (_, inputs, targets) in enumerate(test_loader):
    for j in inputs:
      diff_dict['_'.join([f'{x}' for x in j])] = -1


  for i in range(101):
    print(f'eval diff: {i}%')
    tmp_model = copy.deepcopy(model._network)

    import pdb; pdb.set_trace()
    parameters_to_prune = (
      (tmp_model.backbone.net[0].lw, 'weight'),
      (tmp_model.backbone.net[1].lw, 'weight'),
      (tmp_model.fc, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=i/100,
    )

    for i, (_, inputs, targets) in enumerate(test_loader):

      inputs, targets = inputs.to(model._device), targets.to(model._device)
      logits = tmp_model(inputs)["logits"]

      _, preds = torch.max(logits, dim=1)
      local_correct = preds.eq(targets.expand_as(preds)) # .cpu().sum()

      for idx, j in enumerate(local_correct):
        if j:
          diff_dict['_'.join([f'{x}' for x in inputs[idx]])] = i

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
        uid = '_'.join([f'{x}' for x in inputs[idx]])
        if diff_dict[uid] != -1:
          score += diff_dict[uid]
          cnt += 1

  logging.info(f'Our quality lost score: {score/cnt}')
