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
  # prevent arguments conflict
  if args["quant_method"]=="LUQ":
    if not args["quantizeFwd"]:
      update_message("LUQ", "quantizeFwd", True)
      args["quantizeFwd"] = True
    if not args["quantizeBwd"]:
      update_message("LUQ", "quantizeBwd", True)
      args["quantizeBwd"] = True
    if args["quantGradRound"] != "stoch":
      update_message("LUQ", "quantGradRound", "stoch")
      args["quantGradRound"] = "stoch" 
    if args["quantCalibrate"]:
      update_message("LUQ", "quantCalibrate", "max")
      args["quantCalibrate"] = "max"
    if args["quantizeTrack"]:
      update_message("LUQ", "quantizeTrack", False)
      args["quantizeTrack"] = False
  elif args["quant_method"]=="ours":
    # TODO: add our quantization method
    pass
  if args["quantizeTrack"]:
    # switching both off removes all quantization
    if args["quantizeFwd"]:
      update_message("quantizeTrack", "quantizeFwd", False)
      args["quantizeFwd"] = False
    if args["quantizeBwd"]:
      update_message("quantizeTrack", "quantizeBwd", False)
      args["quantizeBwd"] = False
  return args

def setup_parser():
  parser = argparse.ArgumentParser(
      description='Reproduce of multiple continual learning algorthms.')

  parser.add_argument('--dataset', type=str, default="cifar100")
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
  parser.add_argument('--quantGradRound', type=str, default="standard",
                      required=False, choices=['stoch', 'SQ', 'standard'])
  parser.add_argument('--quantCalibrate', type=str, default="max",
                      required=False, choices=['max'])
  parser.add_argument('--quantizeTrack', action="store_true")
  parser.add_argument('--quant_method', '-qmethod', type=str,
                      default=None, required=False, choices=['LUQ', "IBM",
                                                             'ours'])

  return parser


if __name__ == '__main__':
  main()
