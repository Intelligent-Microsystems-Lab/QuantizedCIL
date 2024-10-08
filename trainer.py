import copy
import datetime
import json
import logging
import os
import sys
import time
import random
import numpy as np

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import ConfigEncoder, count_parameters, save_fc, save_model

import quant
from luq import LUQ
from example_difficulty import determine_difficulty, what_did_i_forget, class_acc, class_acc_diff

# torch.autograd.set_detect_anomaly(True)

def train(args):
  seed_list = copy.deepcopy(args["seed"])
  device = copy.deepcopy(args["device"])

  for seed in seed_list:
    args["seed"] = seed
    args["device"] = device
    _train(args)


def _train(args):
  time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
  args['time_str'] = time_str

  init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
  exp_name = "{}_{}_{}_{}_B{}_Inc{}".format(
      args["time_str"],
      args["dataset"],
      args["model_type"],
      args["seed"],
      init_cls,
      args["increment"],
  )
  args['exp_name'] = exp_name

  if args['debug']:
    logfilename = "logs/debug/{}/{}/{}/{}".format(
        args["prefix"],
        args["dataset"],
        args["model_name"],
        args["exp_name"]
    )
  else:
    logfilename = "logs/{}/{}/{}/{}".format(
        args["prefix"],
        args["dataset"],
        args["model_name"],
        args["exp_name"]
    )

  args['logfilename'] = logfilename

  csv_name = "{}_{}_{}_B{}_Inc{}".format(
      args["dataset"],
      args["seed"],
      args["model_type"],
      init_cls,
      args["increment"],
  )
  args['csv_name'] = csv_name
  os.makedirs(logfilename, exist_ok=True)

  log_path = os.path.join(args["logfilename"], "main.log")
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(filename)s] => %(message)s",
      handlers=[
          logging.FileHandler(filename=log_path),
          logging.StreamHandler(sys.stdout),
      ],
  )

  logging.info(f"Time Str >>> {args['time_str']}")
  # save config
  config_filepath = os.path.join(args["logfilename"], 'configs.json')
  with open(config_filepath, "w") as fd:
    json.dump(args, fd, indent=2, sort_keys=True, cls=ConfigEncoder)

  _set_random(args["seed"])
  _set_device(args)
  print_args(args)
  data_manager = DataManager(
      args["dataset"],
      args["shuffle"],
      args["seed"],
      args["init_cls"],
      args["increment"],
  )
  model = factory.get_model(args["model_name"], args)

  cnn_curve, nme_curve, no_nme = {"top1": [],
                                  "top5": []}, {"top1": [], "top5": []}, True

  # quant overhead
  quant.quantCalibrate = args["quantCalibrate"]
  quant.quantTrack = args["quantizeTrack"]
  quant.quantBits = args["quantBits"]
  quant.quantAccBits = args["quantAccBits"]
  quant.exponent_bits_acc = args["exp_bits_acc"]
  quant.mantissa_bits_acc = args["mant_bits_acc"]
  quant.quantAccFWD = args["quantAccFWD"]
  quant.quantAccBWD = args["quantAccBWD"]
  LUQ.quantAccBits = args["quantAccBits"]
  LUQ.quantAccFWD = args["quantAccFWD"]
  quant.quantMethod = args["quantMethod"]
  quant.quantFWDWgt = args["quantFWDWgt"]
  quant.quantFWDAct = args["quantFWDAct"]
  quant.quantBWDWgt = args["quantBWDWgt"]
  quant.quantBWDAct = args["quantBWDAct"]
  quant.quantBWDGrad1 = args["quantBWDGrad1"]
  quant.quantBWDGrad2 = args["quantBWDGrad2"]
  quant.quantBlockSize = args["quantBlockSize"]
  quant.quantUpdateLowThr = args["quantUpdateLowThr"]
  quant.quantUpdateHighThr = args["quantUpdateHighThr"]
  quant.quantHadOff = args["quantHadOff"]
  quant.quantRequantize = args["quantRequantize"]
  quant.global_args = args
  LUQ.global_args = args

  if 'fp130' in args['quantMethod']:
    quant.quantFP134_rep = '11111110'

  start_time = time.time()
  logging.info(f"Start time:{start_time}")

  for task in range(data_manager.nb_tasks):
    if task == 0 and quant.quantCalibrate:
      quant.calibrate_phase = True
    else:
      quant.calibrate_phase = False
    logging.info("All params: {}".format(count_parameters(model._network)))
    logging.info(
        "Trainable params: {}".format(count_parameters(model._network, True))
    )

    model.incremental_train(data_manager)
    if task == data_manager.nb_tasks - 1:
      cnn_accy, nme_accy = model.eval_task(save_conf=True)
      no_nme = True if nme_accy is None else False
    else:
      cnn_accy, nme_accy = model.eval_task(save_conf=False)
    model.after_task()

    if nme_accy is not None:
      logging.info("CNN: {}".format(cnn_accy["grouped"]))
      logging.info("NME: {}".format(nme_accy["grouped"]))

      cnn_curve["top1"].append(cnn_accy["top1"])
      cnn_curve["top5"].append(cnn_accy["top5"])

      nme_curve["top1"].append(nme_accy["top1"])
      nme_curve["top5"].append(nme_accy["top5"])

      logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
      logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
      logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
      logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
      logging.info(f"Forgetting CNN: {model.forgetting.forgetting_scores}")
      logging.info(f"Forgetting NME: {model.forgetting_nme.forgetting_scores}")
    else:
      logging.info("No NME accuracy.")
      logging.info("CNN: {}".format(cnn_accy["grouped"]))

      cnn_curve["top1"].append(cnn_accy["top1"])
      cnn_curve["top5"].append(cnn_accy["top5"])

      logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
      logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
      logging.info(f"Forgetting CNN: {model.forgetting.forgetting_scores}")
    

  end_time = time.time()
  logging.info(f"End Time:{end_time}")
  cost_time = end_time - start_time
  save_time(args, cost_time)
  save_results(args, cnn_curve, nme_curve, no_nme)

  if args["grad_track"] and args["model_name"] == "icarl":
    try:
      np.save(f"logs/{args['dataset']}/{args['model_name']}/grads/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}_batch.npy",
                quant.grad_track_batch)
      np.save(f"logs/{args['dataset']}/{args['model_name']}/loss/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}_batch.npy",
                quant.loss_track_batch)
      np.save(f"logs/{args['dataset']}/{args['model_name']}/grads/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}_epoch.npy",
                quant.grad_track_epoch)
      np.save(f"logs/{args['dataset']}/{args['model_name']}/loss/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}_epoch.npy",
                quant.loss_track_epoch)
    except:
      os.makedirs(f"logs/{args['dataset']}/{args['model_name']}/grads/", exist_ok=True)
      os.makedirs(f"logs/{args['dataset']}/{args['model_name']}/loss/", exist_ok=True)
      np.save(f"logs/{args['dataset']}/{args['model_name']}/grads/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.grad_track)
      np.save(f"logs/{args['dataset']}/{args['model_name']}/loss/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.loss_track)

  if args["rec_weights"]:
    try:
      np.save(f"logs/{args['dataset']}/{args['model_name']}/weights/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.weight_recording)
    except:
      os.makedirs(f"logs/{args['dataset']}/{args['model_name']}/weights/", exist_ok=True)
      np.save(f"logs/{args['dataset']}/{args['model_name']}/weights/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.weight_recording)
  if args["rec_grads"]:
    try:
      if args['quantMethod'] == "ours":
        np.save(f"logs/{args['dataset']}/{args['model_name']}/bin_usage/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.quant_bin_use_hist)
        np.save(f"logs/{args['dataset']}/{args['model_name']}/under_overflow/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.scale_library_hist)
        np.save(f"logs/{args['dataset']}/{args['model_name']}/gradients/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.gradient_library)
      elif "luq" in args['quantMethod']:
        np.save(f"logs/{args['dataset']}/{args['model_name']}/gradients/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                LUQ.gradient_library)

    except:
      os.makedirs(f"logs/{args['dataset']}/{args['model_name']}/bin_usage/", exist_ok=True)
      os.makedirs(f"logs/{args['dataset']}/{args['model_name']}/under_overflow/", exist_ok=True)
      os.makedirs(f"logs/{args['dataset']}/{args['model_name']}/gradients/", exist_ok=True)
      if args['quantMethod'] == "ours":
        np.save(f"logs/{args['dataset']}/{args['model_name']}/bin_usage/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.quant_bin_use_hist)
        np.save(f"logs/{args['dataset']}/{args['model_name']}/under_overflow/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy", quant.scale_library_hist)
        np.save(f"logs/{args['dataset']}/{args['model_name']}/gradients/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                quant.gradient_library)
      elif "luq" in args['quantMethod']:
        np.save(f"logs/{args['dataset']}/{args['model_name']}/gradients/{args['model_type']}_{args['quantMethod']}_accbits_{args['quantAccBits']}_{args['seed']}.npy",
                LUQ.gradient_library)
  save_model(args, model)

  # import pdb; pdb.set_trace()

def _set_device(args):
  device_type = args["device"]
  gpus = []

  for device in device_type:
    if device == -1:
      device = torch.device("cpu")
    else:
      device = torch.device("cuda") # torch.device("cuda:{}".format(device))

    gpus.append(device)
  args["device"] = gpus
  

def _set_random(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def print_args(args):
  for key, value in args.items():
    logging.info("{}: {}".format(key, value))


def save_time(args, cost_time):
  _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
  os.makedirs(_log_dir, exist_ok=True)
  _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
  with open(_log_path, "a+") as f:
    f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")


def save_results(args, cnn_curve, nme_curve, no_nme=False):
  cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve['top5']
  nme_top1, nme_top5 = nme_curve["top1"], nme_curve['top5']

  # -------CNN TOP1----------
  _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
  os.makedirs(_log_dir, exist_ok=True)

  _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
  if args['prefix'] == 'benchmark':
    with open(_log_path, "a+") as f:
      f.write(f"{args['time_str']},{args['model_name']},")
      for _acc in cnn_top1[:-1]:
        f.write(f"{_acc},")
      f.write(f"{cnn_top1[-1]} \n")
  else:
    assert args['prefix'] in ['fair', 'auc']
    with open(_log_path, "a+") as f:
      f.write(
          f"{args['time_str']},{args['model_name']},{args['memory_size']},")
      for _acc in cnn_top1[:-1]:
        f.write(f"{_acc},")
      f.write(f"{cnn_top1[-1]} \n")

  # -------CNN TOP5----------
  _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top5")
  os.makedirs(_log_dir, exist_ok=True)
  _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
  if args['prefix'] == 'benchmark':
    with open(_log_path, "a+") as f:
      f.write(f"{args['time_str']},{args['model_name']},")
      for _acc in cnn_top5[:-1]:
        f.write(f"{_acc},")
      f.write(f"{cnn_top5[-1]} \n")
  else:
    assert args['prefix'] in ['auc', 'fair']
    with open(_log_path, "a+") as f:
      f.write(
          f"{args['time_str']},{args['model_name']},{args['memory_size']},")
      for _acc in cnn_top5[:-1]:
        f.write(f"{_acc},")
      f.write(f"{cnn_top5[-1]} \n")

  # -------NME TOP1----------
  if no_nme is False:
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
      with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},")
        for _acc in nme_top1[:-1]:
          f.write(f"{_acc},")
        f.write(f"{nme_top1[-1]} \n")
    else:
      assert args['prefix'] in ['fair', 'auc']
      with open(_log_path, "a+") as f:
        f.write(
            f"{args['time_str']},{args['model_name']},{args['memory_size']},")
        for _acc in nme_top1[:-1]:
          f.write(f"{_acc},")
        f.write(f"{nme_top1[-1]} \n")

    # -------NME TOP5----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top5")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
      with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},")
        for _acc in nme_top5[:-1]:
          f.write(f"{_acc},")
        f.write(f"{nme_top5[-1]} \n")
    else:
      assert args['prefix'] in ['auc', 'fair']
      with open(_log_path, "a+") as f:
        f.write(
            f"{args['time_str']},{args['model_name']},{args['memory_size']},")
        for _acc in nme_top5[:-1]:
          f.write(f"{_acc},")
        f.write(f"{nme_top5[-1]} \n")
