import json
import argparse
from trainer import train


def main():
  args = setup_parser().parse_args()
  args = eval_args(args)
  args.config = f"./exps/{args.model_name}.json"
  param = load_json(args.config)
  args = vars(args)  # Converting argparse Namespace to a dict.
  param.update(args)
  print(param)
  train(param)


def load_json(settings_path):
  with open(settings_path) as data_file:
    param = json.load(data_file)
  return param


def update_message(method, argname, value):
  print(f"Update {argname} to {value} because {method} is used.")


def eval_args(args):

  args.dataset = args.dataset.lower()
  if args.dataset == "dsads":
    args.in_dim = 405
  elif args.dataset == "pamap":
    args.in_dim = 243
  elif args.dataset == "hapt":
    args.in_dim = 561
  elif args.dataset == "wisdm":
    args.in_dim = 91
  elif args.dataset == "mnist":
    args.in_dim = 784
  else:
    pass
  return args


def setup_parser():
  parser = argparse.ArgumentParser(
      description='Reproduce of multiple continual learning algorthms.')

  parser.add_argument('--dataset', type=str, default="cifar100")
  parser.add_argument('--in_dim', type=int, default=0)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--memory_size', '-ms', type=int, default=2000)
  parser.add_argument('--init_cls', '-init', type=int, default=10)
  parser.add_argument('--increment', '-incre', type=int, default=10)
  parser.add_argument('--model_name', '-model', type=str,
                      default=None, required=True)
  parser.add_argument('--model_type', '-net', type=str, default='resnet32')
  parser.add_argument('--bias', type=bool, default=False)
  parser.add_argument('--optimizer', '-opt', type=str, default='sgd', 
                      choices=['sgd', 'ours'])
  parser.add_argument('--prefix', '-p', type=str, help='exp type',
                      default='benchmark', choices=['benchmark', 'fair',
                                                    'auc'])
  parser.add_argument('--device', '-d', nargs='+',
                      type=int, default=[0,])
  parser.add_argument('--debug', action="store_true")
  parser.add_argument('--skip', action="store_true",)
  parser.add_argument('--seed', '-seed', nargs='+', type=int, default=[1994],)

  # quant parameters
  parser.add_argument('--quantBits', type=int, default=4)
  parser.add_argument('--quantCalibrate', type=str, default="max",
                      required=False, choices=['max'])
  parser.add_argument('--quantizeTrack', action="store_true")
  parser.add_argument('--quantMethod', '-qmethod', type=str,
                      default=None, required=False, choices=['luq_corrected', 'luq_og', 'fp134', 'fp130',
                                                             'ours', 'noq'])
  parser.add_argument('--quantFWDWgt', '-qfwdw', type=str,
                      default=None, required=False, choices=['sawb', "int",
                                                             'lsq', 'noq', 'mem'])
  parser.add_argument('--quantFWDAct', '-qfwda', type=str,
                      default=None, required=False, choices=['sawb', "int",
                                                             'lsq', 'noq'])
  parser.add_argument('--quantBWDWgt', '-qbwdw', type=str,
                      default=None, required=False, choices=["int", 'noq'])
  parser.add_argument('--quantBWDAct', '-qbwda', type=str,
                      default=None, required=False, choices=["int", 'noq', 'stoch'])
  parser.add_argument('--quantBWDGrad1', '-qbwdg1', type=str, default="int",
                      required=False, choices=['stoch', 'sq', 'int', 'noq'])
  parser.add_argument('--quantBWDGrad2', '-qbwdg2', type=str, default="int",
                      required=False, choices=['stoch', 'sq', 'int', 'noq'])
  parser.add_argument('--quantBlockSize', type=int, default=32)
  parser.add_argument('--quantUpdateP', '-qUP', type=int, default=100)
  parser.add_argument('--quantUpdateLowThr', '-qULT', type=float, default=.7)
  parser.add_argument('--quantUpdateHighThr', '-qUHT', type=float, default=.3)
  parser.add_argument('--quantReplaySize', '-qRS', type=int, default=0,)
  parser.add_argument('--quantHadOff', action="store_true")
  parser.add_argument('--quantAccBits', type=int, default=16)
  parser.add_argument('--quantRequantize', type=bool, default=True)
  parser.add_argument('--LUQCorr', action="store_true")

  # training parameters
  parser.add_argument('--init_epoch', type=int, default=170)
  parser.add_argument('--init_lr', type=float, default=0.05)
  parser.add_argument('--init_milestones', nargs='+', default=[60, 100, 140])
  parser.add_argument('--init_lr_decay', type=float, default=0.1)
  parser.add_argument('--init_weight_decay', type=float,
                      default=2e-4)  # 0.0005
  parser.add_argument('--epochs', type=int, default=170)
  parser.add_argument('--lr', type=float, default=0.05)
  parser.add_argument('--milestones', nargs='+', default=[60, 100, 140])
  parser.add_argument('--lr_decay', type=float, default=0.1)
  parser.add_argument('--weight_decay', type=float, default=2e-4)
  parser.add_argument('--num_workers', type=int, default=8)
  parser.add_argument('--T', type=int, default=2)
  parser.add_argument('--lamda', type=float, default=3)
  parser.add_argument('--split_ratio', type=float, default=0.1)
  parser.add_argument('--fc_hid_dim', type=int, default=100)
  parser.add_argument('--fc_nr_hid', type=int, default=0)
  parser.add_argument('--half_dims', action="store_true")

  # hyperparameter tuning
  # parser.add_argument('--init_dyn_scale', type=float, default=1.1)
  parser.add_argument('--dyn_scale', type=float, default=2.0)
  parser.add_argument('--quantile', type=float, default=0.975)

  parser.add_argument('--example_difficulty', action="store_true")

  parser.add_argument('--fp130', action="store_true")

  return parser


if __name__ == '__main__':
  main()
