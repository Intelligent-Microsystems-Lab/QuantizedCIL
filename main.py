import json
import argparse
from trainer import train


def main():
  args = setup_parser().parse_args()
  # args = eval_args(args)
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


# def eval_args(args):
#   # prevent arguments conflict
#   if args.quantMethod == "luq":
#     if not args.quantizeFwd:
#       update_message("luq", "quantizeFwd", True)
#       args.quantizeFwd = True
#     if not args.quantizeBwd:
#       update_message("luq", "quantizeBwd", True)
#       args.quantizeBwd = True
#     if args.quantGradRound != "stoch":
#       update_message("luq", "quantGradRound", "stoch")
#       args.quantGradRound = "stoch" 
#     if args.quantCalibrate:
#       update_message("luq", "quantCalibrate", "max")
#       args.quantCalibrate = "max"
#     if args.quantizeTrack:
#       update_message("luq", "quantizeTrack", False)
#       args.quantizeTrack = False
#   elif args.quantMethod == "ours":
#     # TODO: add our quantization method
#     if not args.quantizeFwd:
#       update_message("ours", "quantizeFwd", True)
#       args.quantizeFwd = True
#     if not args.quantizeBwd:
#       update_message("ours", "quantizeBwd", True)
#       args.quantizeBwd = True
#     if args.quantCalibrate:
#       update_message("ours", "quantCalibrate", "max")
#       args.quantCalibrate = "max"
#     # if args.quantGradRound != "standard":
#     #   update_message("ours", "quantGradRound", "standard")
#     #   args.quantGradRound = "standard" 
#     if args.quantizeTrack:
#       update_message("ours", "quantizeTrack", False)
#       args.quantizeTrack = False
#   if args.quantizeTrack:
#     # switching both off removes all quantization
#     if args.quantizeFwd:
#       update_message("quantizeTrack", "quantizeFwd", False)
#       args.quantizeFwd = False
#     if args.quantizeBwd:
#       update_message("quantizeTrack", "quantizeBwd", False)
#       args.quantizeBwd = False
#   return args


def setup_parser():
  parser = argparse.ArgumentParser(
      description='Reproduce of multiple continual learning algorthms.')

  parser.add_argument('--dataset', type=str, default="cifar100")
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--memory_size', '-ms', type=int, default=2000)
  parser.add_argument('--init_cls', '-init', type=int, default=10)
  parser.add_argument('--increment', '-incre', type=int, default=10)
  parser.add_argument('--model_name', '-model', type=str,
                      default=None, required=True)
  parser.add_argument('--model_type', '-net', type=str, default='resnet32')
  parser.add_argument('--prefix', '-p', type=str, help='exp type',
                      default='benchmark', choices=['benchmark', 'fair',
                                                    'auc'])
  parser.add_argument('--device', '-d', nargs='+',
                      type=int, default=[0, 1, 2, 3])
  parser.add_argument('--debug', action="store_true")
  parser.add_argument('--skip', action="store_true",)
  parser.add_argument('--seed', '-seed', nargs='+', type=int, default=[1994],)
  parser.add_argument('--quantBits', type=int, default=4)
  parser.add_argument('--quantizeFwd', action="store_true")
  parser.add_argument('--quantizeBwd', action="store_true")
  parser.add_argument('--quantCalibrate', type=str, default="max",
                      required=False, choices=['max'])
  parser.add_argument('--quantizeTrack', action="store_true")
  parser.add_argument('--quantMethod', '-qmethod', type=str,
                      default=None, required=False, choices=['luq', "ibm",
                                                             'ours'])
  parser.add_argument('--quantFWDWgt', '-qfwdw', type=str,
                      default=None, required=False, choices=['sawb', "int",
                                                             'lsq', 'noq'])
  parser.add_argument('--quantFWDAct', '-qfwda', type=str,
                      default=None, required=False, choices=['sawb', "int",
                                                             'lsq', 'noq'])
  parser.add_argument('--quantBWDWgt', '-qbwdw', type=str,
                      default=None, required=False, choices=["int", 'noq'])
  parser.add_argument('--quantBWDAct', '-qbwda', type=str,
                      default=None, required=False, choices=["int", 'noq'])
  parser.add_argument('--quantBWDGrad1', '-qbwdg1' , type=str, default="int",
                      required=False, choices=['stoch', 'sq', 'int', 'noq'])
  parser.add_argument('--quantBWDGrad2', '-qbwdg2' , type=str, default="int",
                      required=False, choices=['stoch', 'sq', 'int', 'noq'])
  # training parameters
  parser.add_argument('--init_epoch', type=int, default=170)
  parser.add_argument('--init_lr', type=float, default=0.05)
  parser.add_argument('--init_milestones', nargs='+',
                      type=int, default=[60, 100, 140])
  parser.add_argument('--init_lr_decay', type=float, default=0.1)
  parser.add_argument('--init_weight_decay', type=float,
                      default=2e-4)  # 0.0005
  parser.add_argument('--epochs', type=int, default=170)
  parser.add_argument('--lr', type=float, default=0.05)
  parser.add_argument('--milestones', nargs='+',
                      type=int, default=[60, 100, 140])
  parser.add_argument('--lr_decay', type=float, default=0.1)
  parser.add_argument('--weight_decay', type=float, default=2e-4)
  parser.add_argument('--num_workers', type=int, default=8)
  parser.add_argument('--T', type=int, default=2)
  parser.add_argument('--lamda', type=float, default=3)
  parser.add_argument('--split_ratio', type=float, default=0.1)

  return parser


if __name__ == '__main__':
  main()
