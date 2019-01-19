import os
import torch
import my_dnn_util.util_torch as my_torch

class UNet(torch.nn.Module):
  def __init__(self,nch_out:int,
               path_file: str):
    super(UNet, self).__init__()
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



