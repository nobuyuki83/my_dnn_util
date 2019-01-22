import os
import torch
import my_dnn_util.util_torch as my_torch

class UNet2(torch.nn.Module):
  def __init__(self,nch_out:int,
               path_file: str):
    super(UNet2, self).__init__()
    self.path_file = path_file
    self.npix = 16
    self.nstride = 2
    #####
    self.layer1 = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=8, padding=3, stride=2), # 1/2
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(inplace=True),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),
      torch.nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1),  # 1/2
      my_torch.ResUnit_BRC_Btl(64),
      my_torch.ResUnit_BRC_Btl(64),
    ) # out: 1/4(64)
    self.layer2 = torch.nn.Sequential( # 1/4(64)
      my_torch.ResUnit_BRC_ResHalf(64, 128),
      my_torch.ResUnit_BRC_Btl(128),
      my_torch.ResUnit_BRC_Btl(128),
    ) # out: 1/8(128)
    self.layer3 = torch.nn.Sequential( # in: 1/8(128)
      my_torch.ResUnit_BRC_ResHalf(128, 256),
      my_torch.ResUnit_BRC_Btl(256),
      my_torch.ResUnit_BRC_Btl(256),
      my_torch.ResUnit_BRC_Btl(256),
      my_torch.ResUnit_BRC_ResDouble(256,128),
      my_torch.ResUnit_BRC_Btl(128),
      my_torch.ResUnit_BRC_Btl(128),
    ) # out: 1/8(128)
    self.layer4 = torch.nn.Sequential( # in 1/8(128)
      my_torch.ResUnit_BRC_ResDouble(256,64),
      my_torch.ResUnit_BRC_Btl(64),
      my_torch.ResUnit_BRC_Btl(64),
    ) # in 1/4(64)
    self.layer5 = torch.nn.Sequential( # in 1/4(128)
      my_torch.ResUnit_BRC_ResDouble(128,64),
      my_torch.ResUnit_BRC_Btl(64),
      my_torch.ResUnit_BRC_Btl(64),
      torch.nn.Conv2d(64, nch_out, kernel_size=1, stride=1),
      torch.nn.Sigmoid()
    ) # out 1/2(3)
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x):
    x1 = self.layer1(x)  # 1/1(3) -> 1/4(64)
    x2 = self.layer2(x1) # 1/4(64) -> 1/8(128)
    x3 = self.layer3(x2) # 1/8(128) -> 1/8(128)
    x4 = self.layer4(torch.cat([x2,x3],1)) # 1/8(128+128) -> 1/4(64)
    x5 = self.layer5(torch.cat([x1,x4],1)) # 1/4(64+64) -> 1/2
    return x5



class UNet1(torch.nn.Module):
  def __init__(self,nch_out:int,
               path_file: str):
    super(UNet1, self).__init__()
    self.path_file = path_file
    self.npix = 16
    self.nstride = 1
    #####
    self.layer0 = torch.nn.Sequential( # 1/1(3)
      torch.nn.Conv2d(3, 32, kernel_size=4, padding=1, stride=2), # 1/2
      my_torch.ResUnit_BRC_Btl(32, True),
      my_torch.ResUnit_BRC_Btl(32, True)
    )  #1/2(32)
    self.layer1 = torch.nn.Sequential( # 1/2(3)
      my_torch.ResUnit_BRC_ResHalf_Cat(32, 64,is_separable=True),
      my_torch.ResUnit_BRC_Btl(64, True),
      my_torch.ResUnit_BRC_Btl(64, True),
    ) # out: 1/4(64)
    self.layer2 = torch.nn.Sequential( # 1/4(64)
      my_torch.ResUnit_BRC_ResHalf_Cat(64, 128,is_separable=True),
      my_torch.ResUnit_BRC_Btl(128, True),
      my_torch.ResUnit_BRC_Btl(128, True),
    ) # out: 1/8(128)
    self.layer3 = torch.nn.Sequential( # in: 1/8(128)
      my_torch.ResUnit_BRC_ResHalf_Cat(128, 256,is_separable=True),
      my_torch.ResUnit_BRC_Btl(256, True),
      my_torch.ResUnit_BRC_Btl(256, True),
      my_torch.ResUnit_BRC_Btl(256, True),
      my_torch.ResUnit_BRC_ResDouble_Cat(256,128,is_separable=True),
      my_torch.ResUnit_BRC_Btl(128, True),
      my_torch.ResUnit_BRC_Btl(128, True),
    ) # out: 1/8(128)
    self.layer4 = torch.nn.Sequential( # in 1/8(128)
      my_torch.ResUnit_BRC_ResDouble_Cat(128+128,128,is_separable=True),
      my_torch.ResUnit_BRC_Btl(128, True),
      my_torch.ResUnit_BRC_Btl(128, True)
    ) # out 1/4(128)
    self.layer5 = torch.nn.Sequential( # in 1/4(128)
      my_torch.ResUnit_BRC_ResDouble_Cat(128+64,64,is_separable=True),
      my_torch.ResUnit_BRC_Btl(64, True),
      my_torch.ResUnit_BRC_Btl(64, True),
    ) # out 1/2(64)
    self.layer6 = torch.nn.Sequential( # in 1/2(128)
      my_torch.ResUnit_BRC_ResDouble_Cat(64+32, 32,is_separable=True),
      my_torch.ResUnit_BRC_Btl(32, True),
      my_torch.ResUnit_BRC_Btl(32, True),
    ) # in 1/1(32)
    self.layer7 = torch.nn.Sequential( # in 1/1(128)
      torch.nn.BatchNorm2d(32+3),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(32+3, 32, kernel_size=1, padding=0, stride=1),
      my_torch.ResUnit_BRC_Btl(32, True),
      my_torch.ResUnit_BRC_Btl(32, True),
      torch.nn.BatchNorm2d(32),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(32, nch_out, kernel_size=1, stride=1),
      torch.nn.Sigmoid()
    ) # out 1/1(3)
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x0):
    x1 = self.layer0(x0)  # 1/1(3) -> 1/2(32)
    x2 = self.layer1(x1)  # 1/2(32) -> 1/4(64)
    x3 = self.layer2(x2) # 1/4(64) -> 1/8(128)
    x4 = self.layer3(x3) # 1/8(128) -> 1/8(128)
    x5 = self.layer4(torch.cat([x3,x4],1)) # 1/8(128+128) -> 1/4(128)
    x6 = self.layer5(torch.cat([x2,x5],1)) # 1/4(64+128) -> 1/2(64)
    x7 = self.layer6(torch.cat([x1,x6],1)) # 1/4(32+64) -> 1/2(16)
    x8 = self.layer7(torch.cat([x0,x7],1)) # 1/2(3+16) -> 1/1(nch_out)
    return x8