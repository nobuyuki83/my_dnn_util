import os
import torch
import torch.nn.functional as F

import my_dnn_util.util_torch as my_torch


class NetDiscriminator(torch.nn.Module):
  def __init__(self,
               nch_in: int,
               path_file: str) -> None:
    super(NetDiscriminator, self).__init__()
    self.path_file = path_file
    self.nstride = 32
    self.layer = torch.nn.Sequential(
      my_torch.ModuleCBR_Half_k4s2( nch_in,  64, is_leaky=True, bn=False),  # 1/2
      my_torch.ModuleCBR_Half_k4s2( 64, 128, is_leaky=True), # 1/4
      my_torch.ModuleCBR_Half_k4s2(128, 256, is_leaky=True), # 1/8
      my_torch.ModuleCBR_Half_k4s2(256, 512, is_leaky=True), # 1/16
      my_torch.ModuleCBR_Half_k4s2(512, 512, is_leaky=True),  # 1/32
      torch.nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1)
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x0):
    x1 = self.layer(x0)
    x2 = torch.sigmoid(x1)
    return x2

class NetDiscriminator256(torch.nn.Module):
  def __init__(self, path_file: str):
    super(NetDiscriminator256, self).__init__()
    self.path_file = path_file
    self.nstride = 32
    self.layer0 = torch.nn.Sequential(
      my_torch.ModuleCBR_k4s2(  3,  64, is_leaky=True, bn=False),  # 1/2
      my_torch.ModuleCBR_k4s2( 64, 128, is_leaky=True), # 1/4
      my_torch.ModuleCBR_k4s2(128, 256, is_leaky=True), # 1/8
      my_torch.ModuleCBR_k4s2(256, 512, is_leaky=True), # 1/16
      my_torch.ModuleCBR_k4s2(512, 512, is_leaky=True),  # 1/32
      my_torch.ModuleCBR_k4s2(512, 512, is_leaky=True),  # 1/64
    )
    self.layer1 = torch.nn.Sequential(
      torch.nn.Linear(512*4*4,1),
      torch.nn.Sigmoid()
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    ####
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x0):
    assert len(x0.shape) == 4
    assert list(x0.shape[1:]) == [3,256,256]
    x1 = self.layer0(x0)
    x1 = x1.view((x0.shape[0],-1))
    x2 = self.layer1(x1)
    return x2

class NetDiscriminator256_A(torch.nn.Module):
  def __init__(self, path_file: str):
    super(NetDiscriminator256_A, self).__init__()
    self.path_file = path_file
    self.nstride = 32
    self.layer0 = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=4, padding=1, stride=2), # 1/2
      my_torch.ResUnit_BRC_ResHalf_Add_B( 64, is_leaky=True, is_separable=True),  # 1/4
      my_torch.ResUnit_BRC_ResHalf_Add_B(128, is_leaky=True, is_separable=True),  # 1/8
      my_torch.ResUnit_BRC_ResHalf_Add_B(256, is_leaky=True, is_separable=True),  # 1/16
      my_torch.ResUnit_BRC_ResHalf_Add_C(512, is_leaky=True, is_separable=True),  # 1/32
      torch.nn.BatchNorm2d(512),
      torch.nn.LeakyReLU(inplace=True,negative_slope=0.2),
      torch.nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1),
      torch.nn.Sigmoid()
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    ####
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x0):
    assert len(x0.shape) == 4
    assert list(x0.shape[1:]) == [3,256,256]
    x1 = self.layer0(x0)
    return x1
