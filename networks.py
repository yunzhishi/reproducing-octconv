import torch.nn as nn
# from octconv_old import OctConv2d
# from octconv_poli import OctConv2d
from octconv_pypi import OctConv2d


def _common_fc():
  """An FC layers group used commonly by either OctConv network and the 
  conventional Conv network. The parameters are carefully calculated, do not
  change unless you know what you are doing.
  """
  return nn.Sequential(nn.Linear(6272, 256),
                       nn.Dropout(0.5),
                       nn.Linear(256, 10))


class OctReLU(nn.Module):
  """ReLU function that should come after any OctConv layer with high- and low-
  frequency outputs.
  """
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.relu_h = nn.ReLU(*args, **kwargs)
    self.relu_l = nn.ReLU(*args, **kwargs)

  def forward(self, x):
    h, l = x
    return self.relu_h(h), self.relu_l(l)


class OctMaxPool2d(nn.Module):
  """MaxPool2d function that should come after any OctConv layer with high- and
  low-frequency outputs.
  """
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.maxpool_h = nn.MaxPool2d(*args, **kwargs)
    self.maxpool_l = nn.MaxPool2d(*args, **kwargs)

  def forward(self, x):
    h, l = x
    return self.maxpool_h(h), self.maxpool_l(l)


class OctCNN(nn.Module):
  """OctCNN network that predicts from the FashionMNIST dataset.
  """
  def __init__(self, alpha=0.5):
    print("Using OctCNN (alpha={:.1})...".format(alpha))
    super().__init__()

    self.convs = nn.Sequential(
      OctConv2d(in_channels=1, out_channels=32,
                kernel_size=3, padding=1, alpha=(0, alpha)),
      OctReLU(),
      OctConv2d(in_channels=32, out_channels=64,
                kernel_size=3, padding=1, alpha=alpha),
      OctReLU(),
      OctConv2d(in_channels=64, out_channels=128,
                kernel_size=3, padding=1, alpha=alpha),
      OctReLU(),
      OctMaxPool2d(2),
      OctConv2d(in_channels=128, out_channels=128,
                kernel_size=3, padding=1, alpha=alpha),
      OctReLU(),
      OctConv2d(in_channels=128, out_channels=128,
                kernel_size=3, padding=1, alpha=(alpha, 0)),
      nn.ReLU(),
      nn.MaxPool2d(2) )

    self.fc = _common_fc()

  def forward(self, x):
    x = self.convs(x)
    x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
    return self.fc(x)


class NormalCNN(nn.Module):
  """CNN network with conventional conv layers that predicts from the
  FashionMNIST dataset.
  """
  def __init__(self):
    print("Using normal CNN...")
    super().__init__()

    self.convs = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=32,
                kernel_size=3, padding=1, bias=False),
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=64,
                kernel_size=3, padding=1, bias=False),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=128,
                kernel_size=3, padding=1, bias=False),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(in_channels=128, out_channels=128,
                kernel_size=3, padding=1, bias=False),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=128,
                kernel_size=3, padding=1, bias=False),
      nn.ReLU(),
      nn.MaxPool2d(2) )

    self.fc = _common_fc()

  def forward(self, x):
    x = self.convs(x)
    x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
    return self.fc(x)
