from models.der import DER
from models.icarl import iCaRL
from models.lwf import LwF
from models.bic import BiC
from models.memo import MEMO


def get_model(model_name, args):
  name = model_name.lower()
  if name == "icarl":
    return iCaRL(args)
  elif name == "bic":
    return BiC(args)
  elif name == "lwf":
    return LwF(args)
  elif name == "der":
      return DER(args)
  elif name == 'memo':
      return MEMO(args)
  else:
    assert 0

