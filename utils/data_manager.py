import logging
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iMNIST, iDSADS, iPAMAP, iHAPT, iWISDM


class DataManager(object):
  def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
    self.dataset_name = dataset_name
    self._setup_data(dataset_name, shuffle, seed)
    assert init_cls <= len(self._class_order), "No enough classes."
    self._increments = [init_cls]
    while sum(self._increments) + increment < len(self._class_order):
      self._increments.append(increment)
    offset = len(self._class_order) - sum(self._increments)
    if offset > 0:
      self._increments.append(offset)

  @property
  def nb_tasks(self):
    return len(self._increments)

  def get_task_size(self, task):
    return self._increments[task]

  def get_total_classnum(self):
    return len(self._class_order)

  def get_dataset(
      self, indices, source, mode, appendent=None, ret_data=False, m_rate=None,
      no_trsf=False,
      debug=False
  ):
    if source == "train":
      x, y = self._train_data, self._train_targets
    elif source == "test":
      x, y = self._test_data, self._test_targets
    else:
      raise ValueError("Unknown data source {}.".format(source))

    if mode == "train":
      trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
    elif mode == "flip":
      trsf = transforms.Compose(
          [
              *self._test_trsf,
              transforms.RandomHorizontalFlip(p=1.0),
              *self._common_trsf,
          ]
      )
    elif mode == "test":
      trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
    else:
      raise ValueError("Unknown mode {}.".format(mode))

    data, targets = [], []
    if debug:
      import pdb; pdb.set_trace()
    for idx in indices:
      if m_rate is None:
        class_data, class_targets = self._select(
            x, y, low_range=idx, high_range=idx + 1
        )
      else:
        class_data, class_targets = self._select_rmm(
            x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
        )
      data.append(class_data)
      targets.append(class_targets)

    if appendent is not None and len(appendent) != 0:
      appendent_data, appendent_targets = appendent
      data.append(appendent_data)
      targets.append(appendent_targets)

    data, targets = np.concatenate(data), np.concatenate(targets)
    # if no_trsf:
    #   trsf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x) )])
    if ret_data:
      return data, targets, DummyDataset(data, targets, trsf, self.use_path,
                                         self.datatype)
    else:
      return DummyDataset(data, targets, trsf, self.use_path, self.datatype)

  def get_dataset_with_split(
      self, indices, source, mode, appendent=None, val_samples_per_class=0
  ):
    if source == "train":
      x, y = self._train_data, self._train_targets
    elif source == "test":
      x, y = self._test_data, self._test_targets
    else:
      raise ValueError("Unknown data source {}.".format(source))

    if mode == "train":
      trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
    elif mode == "test":
      trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
    else:
      raise ValueError("Unknown mode {}.".format(mode))

    train_data, train_targets = [], []
    val_data, val_targets = [], []
    for idx in indices:
      class_data, class_targets = self._select(
          x, y, low_range=idx, high_range=idx + 1
      )
      val_indx = np.random.choice(
          len(class_data), val_samples_per_class, replace=False
      )
      train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
      val_data.append(class_data[val_indx])
      val_targets.append(class_targets[val_indx])
      train_data.append(class_data[train_indx])
      train_targets.append(class_targets[train_indx])

    if appendent is not None:
      appendent_data, appendent_targets = appendent
      for idx in range(0, int(np.max(appendent_targets)) + 1):
        append_data, append_targets = self._select(
            appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
        )
        val_indx = np.random.choice(
            len(append_data), val_samples_per_class, replace=False
        )
        train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
        val_data.append(append_data[val_indx])
        val_targets.append(append_targets[val_indx])
        train_data.append(append_data[train_indx])
        train_targets.append(append_targets[train_indx])

    train_data, train_targets = np.concatenate(train_data), np.concatenate(
        train_targets
    )
    val_data, val_targets = np.concatenate(
        val_data), np.concatenate(val_targets)

    return DummyDataset(
        train_data, train_targets, trsf, self.use_path, self.datatype
    ), DummyDataset(val_data, val_targets, trsf, self.use_path, self.datatype)

  def _setup_data(self, dataset_name, shuffle, seed):
    idata = _get_idata(dataset_name)
    try:
      idata.download_data()
      self.datatype = "image"
    except:
      idata.org_data()
      self.datatype = "sensor"

    # Data
    self._train_data, self._train_targets = idata.train_data, idata.train_targets
    self._test_data, self._test_targets = idata.test_data, idata.test_targets
    self.use_path = idata.use_path

    # Transforms
    self._train_trsf = idata.train_trsf
    self._test_trsf = idata.test_trsf
    self._common_trsf = idata.common_trsf

    # Order
    # order = [i for i in range(len(np.unique(self._train_targets)))]
    #TODO is this correct?
    order = [i for i in np.unique(self._train_targets)]
    if shuffle:
      np.random.seed(seed)
      # order = np.random.permutation(len(order)).tolist()
      order = np.random.permutation(order).tolist()
    else:
      order = idata.class_order
    self._class_order = order
    logging.info(self._class_order)

    # Map indices
    self._train_targets, self.translate_dict = _map_new_class_index(
        self._train_targets, self._class_order
    )
    self._test_targets, _ = _map_new_class_index(
        self._test_targets, self._class_order)

  def _select(self, x, y, low_range, high_range):
    idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
    return x[idxes], y[idxes]

  def _select_rmm(self, x, y, low_range, high_range, m_rate):
    assert m_rate is not None
    if m_rate != 0:
      idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
      selected_idxes = np.random.randint(
          0, len(idxes), size=int((1 - m_rate) * len(idxes))
      )
      new_idxes = idxes[selected_idxes]
      new_idxes = np.sort(new_idxes)
    else:
      new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
    return x[new_idxes], y[new_idxes]

  def getlen(self, index):
    y = self._train_targets
    return np.sum(np.where(y == index))

def reorganize_image_data(data):
  if type(data[0]) == torch.Tensor:
    data = np.array([d.numpy() for d in data])
    if data[0].shape[0] == 3:
      data = np.array([np.transpose(d, (1, 2, 0)) for d in data])
    if type(data[0][0][0][0]) != np.uint8:
      data = np.array([(d*255).astype(np.uint8) for d in data])
  return data

class DummyDataset(Dataset):
  def __init__(self, data, labels, trsf, use_path=False, datatype="image"):
    assert len(data) == len(labels), "Data size error!"
    self.data = data
    self.labels = labels
    self.trsf = trsf
    self.use_path = use_path
    self.datatype = datatype
    if self.datatype == "image":
      self.data = reorganize_image_data(self.data)
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if self.datatype == "image":

      if self.use_path:
        data = self.trsf(pil_loader(self.data[idx]))
      else:
        try:
          data = self.trsf(Image.fromarray(self.data[idx]))
          
        except:
          try:
            data = self.trsf(pil_loader(self.data[idx]))
          except:
            import pdb; pdb.set_trace()
    else:
      data = self.trsf(self.data[idx])
    label = self.labels[idx]

    return idx, data, label


def _map_new_class_index(y, order):
  translate_dict = {order.index(i):i for i in np.unique(y)}
  return np.array(list(map(lambda x: order.index(x), y))), translate_dict


def _get_idata(dataset_name):
  name = dataset_name.lower()
  if name == "cifar10":
    return iCIFAR10()
  elif name == "cifar100":
    return iCIFAR100()
  elif name == "imagenet1000":
    return iImageNet1000()
  elif name == "imagenet100":
    return iImageNet100()
  elif name == "mnist":
    return iMNIST()
  elif name == "dsads":
    return iDSADS()
  elif name == "pamap":
    return iPAMAP()
  elif name == "hapt":
    return iHAPT()
  elif name == "wisdm":
    return iWISDM()
  else:
    raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
  """
  Ref:
  https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
  """
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, "rb") as f:
    img = Image.open(f)
    return img.convert("RGB")


def accimage_loader(path):
  """
  Ref:
  https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
  accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
  accimage is available on conda-forge.
  """
  import accimage

  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  """
  Ref:
  https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
  """
  from torchvision import get_image_backend

  if get_image_backend() == "accimage":
    return accimage_loader(path)
  else:
    return pil_loader(path)
