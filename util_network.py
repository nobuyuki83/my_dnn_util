import os
import torch
import torch.nn.functional
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

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
      my_torch.ModuleBRC_ResBtl(64),
      my_torch.ModuleBRC_ResBtl(64),
    ) # out: 1/4(64)
    self.layer2 = torch.nn.Sequential( # 1/4(64)
      my_torch.ModuleBRC_Half_ResAdd(64, 128),
      my_torch.ModuleBRC_ResBtl(128),
      my_torch.ModuleBRC_ResBtl(128),
    ) # out: 1/8(128)
    self.layer3 = torch.nn.Sequential( # in: 1/8(128)
      my_torch.ModuleBRC_Half_ResAdd(128, 256),
      my_torch.ModuleBRC_ResBtl(256),
      my_torch.ModuleBRC_ResBtl(256),
      my_torch.ModuleBRC_ResBtl(256),
      my_torch.ResUnit_BRC_ResDouble(256,128),
      my_torch.ModuleBRC_ResBtl(128),
      my_torch.ModuleBRC_ResBtl(128),
    ) # out: 1/8(128)
    self.layer4 = torch.nn.Sequential( # in 1/8(128)
      my_torch.ResUnit_BRC_ResDouble(256,64),
      my_torch.ModuleBRC_ResBtl(64),
      my_torch.ModuleBRC_ResBtl(64),
    ) # in 1/4(64)
    self.layer5 = torch.nn.Sequential( # in 1/4(128)
      my_torch.ResUnit_BRC_ResDouble(128,64),
      my_torch.ModuleBRC_ResBtl(64),
      my_torch.ModuleBRC_ResBtl(64),
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


class NetEncDec_s2_A(torch.nn.Module):
  def __init__(self,
               nch_in: int,
               nch_out: int,
               path_file: str):
    super(NetEncDec_s2_A, self).__init__()
    self.path_file = path_file
    self.nstride = 2
    self.npix = 8
    #####
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(nch_in, 64, kernel_size=5, padding=2, stride=1),   # 1/1
      my_torch.ModuleBRC_Half_ResCat( 64, 128, nc_in_group=1),  # 1/2
      my_torch.ModuleBRC_Half_ResCat(128, 256, nc_in_group=1), # 1/4
      my_torch.ModuleBRC_Half_ResCat(256, 512, nc_in_group=1), # 1/8
      my_torch.ModuleBRC_ResBtl(512, nc_in_group=1),
      my_torch.ModuleBRC_ResBtl(512, nc_in_group=1),
      my_torch.ModuleBRC_ResBtl(512, nc_in_group=1),
      my_torch.ModuleBRC_ResBtl(512, nc_in_group=1),
      my_torch.ModuleBRC_ResBtl(512, nc_in_group=1),
      my_torch.ModuleBRC_ResBtl(512, nc_in_group=1),
      my_torch.ModuleBRC_Double_ResCat(512, 256, nc_in_group=1), # 1/4
      my_torch.ModuleBRC_Double_ResCat(256, 128, nc_in_group=1), # 1/2
      torch.nn.BatchNorm2d(128),
      torch.nn.ReLU(),
      torch.nn.Conv2d(128, nch_out, kernel_size=5, padding=2, stride=1),
      torch.nn.Sigmoid()
    ) # out 1/2(3)
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x):
    return self.model(x)

########################################################################################################################
########################################################################################################################
########################################################################################################################

class NetEncDec_s1_A(torch.nn.Module):
  def __init__(self,nch_in:int,
               nch_out:int,
               path_file: str):
    super(NetEncDec_s1_A, self).__init__()
    self.path_file = path_file
    self.nstride = 1
    self.npix = 8
    #####
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(nch_in, 64, kernel_size=5, padding=2, stride=1),   # 1/1
      my_torch.ModuleBRC_Half_ResCat(64, 128, is_separable=True),  # 1/2
      my_torch.ModuleBRC_Half_ResCat(128, 256, is_separable=True), # 1/4
      my_torch.ModuleBRC_Half_ResCat(256, 512, is_separable=True), # 1/8
      my_torch.ModuleBRC_ResBtl(512, is_separable=True),
      my_torch.ModuleBRC_ResBtl(512, is_separable=True),
      my_torch.ModuleBRC_ResBtl(512, is_separable=True),
      my_torch.ModuleBRC_ResBtl(512, is_separable=True),
      my_torch.ModuleBRC_ResBtl(512, is_separable=True),
      my_torch.ModuleBRC_ResBtl(512, is_separable=True),
      my_torch.ModuleBRC_Double_ResCat(512, 256, is_separable=True), # 1/4
      my_torch.ModuleBRC_Double_ResCat(256, 128, is_separable=True), # 1/2
      my_torch.ModuleBRC_Double_ResCat(128, 64, is_separable=True),  # 1/2
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(),
      torch.nn.Conv2d(64, nch_out, kernel_size=5, padding=2, stride=1),
      torch.nn.Sigmoid()
    ) # out 1/2(3)
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x):
    return self.model(x)

class NetEncDec_s1_B(torch.nn.Module):
  def __init__(self,nch_in:int, nch_out:int,
               path_file: str):
    super(NetEncDec_s1_B, self).__init__()
    self.path_file = path_file
    self.npix = 4
    self.nstride = 1
    #####
    self.layer = torch.nn.Sequential( # 1/1(3)
#      torch.nn.Conv2d(nch_in, 64, kernel_size=5, padding=2, stride=1),
      torch.nn.Conv2d(nch_in, 64, kernel_size=3, padding=1, stride=1),
      my_torch.ModuleBRC_Mob(64, nc_in_group=1),
      my_torch.ModuleBRC_Half_ResAdd(64, 128, is_separable=True),
      my_torch.ModuleBRC_Mob(128, nc_in_group=1),
      my_torch.ModuleBRC_Half_ResAdd(128, 256, is_separable=True),
      my_torch.ModuleBRC_Mob(256, nc_in_group=2),
      my_torch.ModuleBRC_Mob(256, nc_in_group=2),
      my_torch.ModuleBRC_Mob(256, nc_in_group=2),
      my_torch.ModuleBRC_Mob(256, nc_in_group=2),
      my_torch.ModuleBRC_Mob(256, nc_in_group=2),
      my_torch.ModuleBRC_Mob(256, nc_in_group=2),
      my_torch.ResUnit_BRC_ResDouble(256,128, is_separable=True),
      my_torch.ModuleBRC_Mob(128, nc_in_group=1),
      my_torch.ResUnit_BRC_ResDouble(128, 64, is_separable=True),
      my_torch.ModuleBRC_Mob(64, nc_in_group=1),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(64, nch_out, kernel_size=5, padding=2, stride=1),
      torch.nn.Sigmoid(),
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x0):
    x1 = self.layer(x0)
    return x1

class NetEncDec_s1_C(torch.nn.Module):
  def __init__(self,nch_in:int, nch_out:int,
               path_file: str):
    super(NetEncDec_s1_C, self).__init__()
    self.path_file = path_file
    self.npix = 4
    self.nstride = 1
    #####
    self.layer = torch.nn.Sequential( # 1/1(3)
      torch.nn.Conv2d(nch_in, 64, kernel_size=7, padding=3, stride=1),
      my_torch.ModuleBRC_k3(64,64),
      my_torch.ModuleBRC_Half_k4s2(64, 128),
      my_torch.ModuleBRC_k3(128,128),
      my_torch.ModuleBRC_Half_k4s2(128, 256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_Double_k4s2(256,128),
      my_torch.ModuleBRC_k3(128,128),
      my_torch.ModuleBRC_Double_k4s2(128, 64),
      my_torch.ModuleBRC_k3(64,64),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(64, nch_out, kernel_size=7, padding=3, stride=1),
      torch.nn.Tanh(),
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x0):
    x1 = self.layer(x0)
    return x1

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
      my_torch.ModuleBRC_ResBtl(32, True),
      my_torch.ModuleBRC_ResBtl(32, True)
    )  #1/2(32)
    self.layer1 = torch.nn.Sequential( # 1/2(3)
      my_torch.ModuleBRC_Half_ResCat(32, 64, is_separable=True),
      my_torch.ModuleBRC_ResBtl(64, True),
      my_torch.ModuleBRC_ResBtl(64, True),
    ) # out: 1/4(64)
    self.layer2 = torch.nn.Sequential( # 1/4(64)
      my_torch.ModuleBRC_Half_ResCat(64, 128, is_separable=True),
      my_torch.ModuleBRC_ResBtl(128, True),
      my_torch.ModuleBRC_ResBtl(128, True),
    ) # out: 1/8(128)
    self.layer3 = torch.nn.Sequential( # in: 1/8(128)
      my_torch.ModuleBRC_Half_ResCat(128, 256, is_separable=True),
      my_torch.ModuleBRC_ResBtl(256, True),
      my_torch.ModuleBRC_ResBtl(256, True),
      my_torch.ModuleBRC_ResBtl(256, True),
      my_torch.ModuleBRC_Double_ResCat(256, 128, is_separable=True),
      my_torch.ModuleBRC_ResBtl(128, True),
      my_torch.ModuleBRC_ResBtl(128, True),
    ) # out: 1/8(128)
    self.layer4 = torch.nn.Sequential( # in 1/8(128)
      my_torch.ModuleBRC_Double_ResCat(128 + 128, 128, is_separable=True),
      my_torch.ModuleBRC_ResBtl(128, True),
      my_torch.ModuleBRC_ResBtl(128, True)
    ) # out 1/4(128)
    self.layer5 = torch.nn.Sequential( # in 1/4(128)
      my_torch.ModuleBRC_Double_ResCat(128 + 64, 64, is_separable=True),
      my_torch.ModuleBRC_ResBtl(64, True),
      my_torch.ModuleBRC_ResBtl(64, True),
    ) # out 1/2(64)
    self.layer6 = torch.nn.Sequential( # in 1/2(128)
      my_torch.ModuleBRC_Double_ResCat(64 + 32, 32, is_separable=True),
      my_torch.ModuleBRC_ResBtl(32, True),
      my_torch.ModuleBRC_ResBtl(32, True),
    ) # in 1/1(32)
    self.layer7 = torch.nn.Sequential( # in 1/1(128)
      torch.nn.BatchNorm2d(32+3),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(32+3, 32, kernel_size=1, padding=0, stride=1),
      my_torch.ModuleBRC_ResBtl(32, True),
      my_torch.ModuleBRC_ResBtl(32, True),
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




class NetEncDec_s1p4_Dilated(torch.nn.Module):
  def __init__(self,nch_in:int, nch_out:int,
               path_file: str,
               af='tanh'):
    super(NetEncDec_s1p4_Dilated, self).__init__()
    self.path_file = path_file
    self.npix = 4
    self.nstride = 1
    #####
    self.layer = torch.nn.Sequential( # 1/1(3)
      torch.nn.Conv2d(nch_in, 64, kernel_size=5, padding=2, stride=1), # 1/2
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(inplace=True),
      my_torch.ModuleCBR_Half_k4s2(64, 128),
      my_torch.ModuleCBR_k3(128, 128),
      my_torch.ModuleCBR_Half_k4s2(128, 256),
      my_torch.ModuleCBR_k3(256, 256),
      my_torch.ModuleCBR_k3(256, 256),
      my_torch.ModuleCBR_k3(256, 256, dilation=2),
      my_torch.ModuleCBR_k3(256, 256, dilation=4),
      my_torch.ModuleCBR_k3(256, 256, dilation=8),
      my_torch.ModuleCBR_k3(256, 256, dilation=16),
      my_torch.ModuleCBR_k3(256, 256),
      my_torch.ModuleCBR_k3(256, 256),
      my_torch.ModuleCBR_Double_k4s2(256,128),
      my_torch.ModuleCBR_k3(128, 128),
      my_torch.ModuleCBR_Double_k4s2(128, 64),
      my_torch.ModuleCBR_k3(64, 32),
      torch.nn.Conv2d(32,nch_out,kernel_size=3, padding=1, stride=1),
    )
    if af == 'tanh':
      self.af1 = torch.nn.Tanh()
    elif af == 'sigmoid':
      self.af1 = torch.nn.Sigmoid()
    else:
      print("unknown activation function",af)
      exit()

    my_torch.initialize_net(self)
    ####
    if os.path.isfile(path_file):
      self = my_torch.load_model_cpm(self, path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def forward(self, x0):
    x1 = self.layer(x0)
    return self.af1(x1)

##################################################################################################


class NetDiscriminator(torch.nn.Module):
  def __init__(self, path_file: str):
    super(NetDiscriminator, self).__init__()
    self.path_file = path_file
    self.nstride = 32
    self.layer = torch.nn.Sequential(
      my_torch.ModuleCBR_k4s2(  3,  64, is_leaky=True, bn=False),  # 1/2
      my_torch.ModuleCBR_k4s2( 64, 128, is_leaky=True), # 1/4
      my_torch.ModuleCBR_k4s2(128, 256, is_leaky=True), # 1/8
      my_torch.ModuleCBR_k4s2(256, 512, is_leaky=True), # 1/16
      my_torch.ModuleCBR_k4s2(512, 512, is_leaky=True),  # 1/32
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

##############################################################################################


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

class Net_VGG16(torch.nn.Module):
  def __init__(self,model_dir):
    super(Net_VGG16, self).__init__()
    print("loading vgg16")
    url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
    vgg_state_dict = model_zoo.load_url(url, model_dir=model_dir)
    vgg_keys = list(vgg_state_dict.keys())

    self.conv1_1 = torch.nn.Conv2d( 3,64,kernel_size=3,stride=1,padding=1)
    self.conv1_2 = torch.nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
    ####
    self.conv2_1 = torch.nn.Conv2d( 64,128,kernel_size=3,stride=1,padding=1)
    self.conv2_2 = torch.nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
    ####
    self.conv3_1 = torch.nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
    self.conv3_2 = torch.nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
    self.conv3_3 = torch.nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
    self.conv3_4 = torch.nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
    ####
#    self.conv4_1 = torch.nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1)
#    self.conv4_2 = torch.nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)


    self.conv1_1.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[0]])
    self.conv1_1.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[1]])
    self.conv1_2.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[2]])
    self.conv1_2.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[3]])
    ####
    self.conv2_1.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[4]])
    self.conv2_1.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[5]])
    self.conv2_2.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[6]])
    self.conv2_2.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[7]])
    ####
    self.conv3_1.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[8]])
    self.conv3_1.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[9]])
    self.conv3_2.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[10]])
    self.conv3_2.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[11]])
    self.conv3_3.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[12]])
    self.conv3_3.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[13]])
    self.conv3_4.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[14]])
    self.conv3_4.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[15]])
    ####
#    self.conv4_1.weight = torch.nn.Parameter(vgg_state_dict[vgg_keys[16]])
#    self.conv4_2.bias = torch.nn.Parameter(vgg_state_dict[vgg_keys[17]])

    if torch.cuda.is_available():
      self = self.cuda()

  def prep256(self,y0):
    assert len(y0.shape) == 4
#    assert tuple(y0.shape[1:]) == (3,256,256)
    assert tuple(y0.shape[1:]) == (3,224,224)
    assert torch.min(y0[:,0,:,:])>-1.1 and  torch.max(y0[:,0,:,:])<+1.1
    assert torch.min(y0[:,1,:,:])>-1.1 and  torch.max(y0[:,1,:,:])<+1.1
    assert torch.min(y0[:,2,:,:])>-1.1 and  torch.max(y0[:,2,:,:])<+1.1
    y0[:, 0, :, :] -= (103.939 / 128.0) - 1.0
    y0[:, 1, :, :] -= (116.779 / 128.0) - 1.0
    y0[:, 2, :, :] -= (123.680 / 128.0) - 1.0
    return y0

  def forward(self,x0):
    assert len(x0.shape) == 4
    assert tuple(x0.shape[1:]) == (3,224,224)

    x1 = torch.nn.functional.relu(self.conv1_1(x0))
    x1 = torch.nn.functional.relu(self.conv1_2(x1))

    x2 = torch.nn.functional.max_pool2d(x1,kernel_size=2,stride=2)
    x2 = torch.nn.functional.relu(self.conv2_1(x2))
    x2 = torch.nn.functional.relu(self.conv2_2(x2))

    x3 = torch.nn.functional.max_pool2d(x2,kernel_size=2,stride=2)
    x3 = torch.nn.functional.relu(self.conv3_1(x3))
    x3 = torch.nn.functional.relu(self.conv3_2(x3))
    x3 = torch.nn.functional.relu(self.conv3_3(x3))
    x3 = torch.nn.functional.relu(self.conv3_4(x3))
    '''
    x4 = torch.nn.functional.max_pool2d(x3,kernel_size=2,stride=2)
    x4 = torch.nn.functional.relu(self.conv4_1(x4))
    x4 = torch.nn.functional.relu(self.conv4_2(x4))
    '''
#    x1 = x1/(x1.shape[1]*x1.shape[2]*x1.shape[3])
#    x2 = x2/(x2.shape[1]*x2.shape[2]*x2.shape[3])
#    x3 = x3/(x3.shape[1]*x3.shape[2]*x3.shape[3])
#    x4 = x4/(x4.shape[1]*x4.shape[2]*x4.shape[3])
#    return x1,x2,x3,x4
    return x1,x2,x3

  def mseloss(self,x0,y0):
#    l0 = torch.nn.functional.mse_loss(x0[0],y0[0])
#    l1 = torch.nn.functional.mse_loss(x0[1],y0[1])
    l2 = torch.nn.functional.mse_loss(x0[2],y0[2])
#    return l0+l1+l2
    return l2
