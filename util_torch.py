import os, numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

def np2pt(np_img,scale,requires_grad=False):
  np_img0 = np_img.view()
  if np_img.ndim == 3:
    np_img0 = np_img.reshape([1]+list(np_img.shape))
  if np_img.ndim == 2:
    np_img0 = np_img.reshape(tuple([1]+list(np_img.shape)+[1]))
  pt_img = torch.from_numpy(numpy.moveaxis(np_img0, 3, 1).astype(numpy.float32)*scale)
  vpt_img = torch.autograd.Variable(pt_img, requires_grad=requires_grad)
  if torch.cuda.is_available():
    vpt_img = vpt_img.cuda()
  return vpt_img

def load_model_cpm(net_cpm,path):
  if os.path.isfile(path):
    print("load model from",path)
    if torch.cuda.is_available():
      net_cpm.load_state_dict(torch.load(path))
    else:
      net_cpm.load_state_dict(torch.load(path, map_location='cpu'))
  return net_cpm

def initialize_net(net):
  for m in net.modules():
    if isinstance(m, torch.nn.Conv2d):
      torch.nn.init.xavier_uniform_(m.weight, 1.414)
      torch.nn.init.constant_(m.bias, 0.1)
    if isinstance(m, torch.nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight, 1.1414)
      torch.nn.init.constant_(m.bias, 0.1)
    if isinstance(m, torch.nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()


class ResUnit_BRC_Btl(torch.nn.Module):
  def __init__(self, nc, is_separable=False):
    super(ResUnit_BRC_Btl, self).__init__()
    assert nc%2 == 0
    nh = nc//2
    ngroup = nh if is_separable else 1
    self.net = torch.nn.Sequential(
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc, nh, kernel_size=1),
      torch.nn.BatchNorm2d(nh),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nh, nh, kernel_size=3, padding=1, groups=ngroup),
      torch.nn.BatchNorm2d(nh),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nh, nc, kernel_size=1),
    )
    initialize_net(self)

  def forward(self, x):
    return self.net(x)+x

class ResUnit_BRC_Mob(torch.nn.Module):
  def __init__(self, nc):
    super(ResUnit_BRC_Mob, self).__init__()
    self.net = torch.nn.Sequential(
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc, nc, kernel_size=1),
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc, nc, kernel_size=3, padding=1, groups=nc),
    )
    initialize_net(self)

  def forward(self, x):
    return self.net(x)+x

class ModuleConv_k3(torch.nn.Module):
  def __init__(self, nc_in, nc_out,dilation=1, is_leaky=False, bn=True):
    super(ModuleConv_k3, self).__init__()
    padding = dilation
    layers = []
    layers.append( torch.nn.Conv2d(nc_in, nc_out, kernel_size=3, stride=1, padding=padding, dilation=dilation) )
    if bn:
      layers.append(torch.nn.BatchNorm2d(nc_out))
    if not is_leaky:
      layers.append(torch.nn.ReLU(inplace=True))
    else:
      layers.append(torch.nn.LeakyReLU(inplace=True,negative_slope=0.2))
    self.model = nn.Sequential(*layers)
    initialize_net(self)

  def forward(self, x):
    return self.model(x)


class ModuleConv_k4_s2(torch.nn.Module):
  def __init__(self, nc_in, nc_out, is_leaky=False, bn=True):
    super(ModuleConv_k4_s2, self).__init__()
    layers = []
    layers.append(torch.nn.Conv2d(nc_in, nc_out, kernel_size=4, padding=1, stride=2))
    if bn:
      layers.append(torch.nn.BatchNorm2d(nc_out))
    if not is_leaky:
      layers.append( torch.nn.ReLU(inplace=True) )
    else:
      layers.append(torch.nn.LeakyReLU(inplace=True, negative_slope=0.2))
    self.model = nn.Sequential(*layers)
    initialize_net(self)

  def forward(self, x):
    return self.model(x)


class ModuleDeconv_k4_s2(torch.nn.Module):
  def __init__(self, nc_in, nc_out):
    super(ModuleDeconv_k4_s2, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(nc_in, nc_out, kernel_size=4, padding=1, stride=2),
      torch.nn.BatchNorm2d(nc_out),
      torch.nn.ReLU(inplace=True),
    )
    initialize_net(self)

  def forward(self, x):
    return self.model(x)


class NetUnit_Res(torch.nn.Module):
  def __init__(self, nc):
    super(NetUnit_Res, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(nc, nc, kernel_size=3, padding=1),
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(),
      torch.nn.Conv2d(nc, nc, kernel_size=3, padding=1),
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(),
    )
    initialize_net(self)

  def forward(self, x):
    return x+self.model(x)


class ResUnit_BRC(torch.nn.Module):
  def __init__(self, nc):
    super(ResUnit_BRC, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1),
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc, nc, kernel_size=3, padding=1, stride=1),
    )
    initialize_net(self)

  def forward(self, x):
    return x+self.model(x)


class ResUnit_BRC_ResHalf(torch.nn.Module):
  def __init__(self, nc_in, nc_out, is_separable=False):
    super(ResUnit_BRC_ResHalf, self).__init__()
    ngroup = nc_out if is_separable else 1
    self.bn1 = torch.nn.BatchNorm2d(nc_in)
    self.conv1 = torch.nn.Conv2d(nc_in, nc_out, kernel_size=1, padding=0, stride=1)
    ###
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(nc_in, nc_out, kernel_size=4, padding=1, stride=2),
      torch.nn.BatchNorm2d(nc_out),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc_out, nc_out, kernel_size=3, padding=1,stride=1,groups=ngroup),
    )
    initialize_net(self)

  def forward(self, x):
    x = torch.nn.functional.relu(self.bn1(x))
    y = torch.nn.functional.max_pool2d(self.conv1(x),kernel_size=4,padding=1,stride=2)
    return y+self.model(x)


class ResUnit_BRC_ResHalf_Cat(torch.nn.Module):
  def __init__(self, nc_in, nc_out, is_separable=False):
    super(ResUnit_BRC_ResHalf_Cat, self).__init__()
    nh0 = nc_out//2
    nh1 = nc_out-nh0
    ngroup = nh0 if is_separable else 1
    ####
    self.bn1 = torch.nn.BatchNorm2d(nc_in)
    self.conv1 = torch.nn.Conv2d(nc_in, nh1, kernel_size=1, padding=0, stride=1)
    ####
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(nc_in, nh0, kernel_size=4, padding=1, stride=2),
      torch.nn.BatchNorm2d(nh0),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nh0, nh0, kernel_size=3, padding=1,stride=1,groups=ngroup),
    )
    initialize_net(self)

  def forward(self, x):
    x = torch.nn.functional.relu(self.bn1(x))
    y = torch.nn.functional.max_pool2d(self.conv1(x),kernel_size=4,padding=1,stride=2)
    return torch.cat((y,self.model(x)),1)


class ResUnit_BRC_ResHalf_Cat_A(torch.nn.Module):
  def __init__(self, nc_in, nc_out, is_separable=False, is_leaky=True, bn=True):
    super(ResUnit_BRC_ResHalf_Cat_A, self).__init__()
    nh = nc_out//2
    assert nc_out == nh*2
    ngroup = nh if is_separable else 1
    ####
    layers = []
    if bn == True:
      layers.append( torch.nn.BatchNorm2d(nc_in) )
    if is_leaky:
      layers.append( torch.nn.LeakyReLU(inplace=True,negative_slope=0.2) )
    else:
      layers.append( torch.nn.ReLU(inplace=True) )
    layers.append( torch.nn.Conv2d(nc_in, nh, kernel_size=4, padding=1, stride=2) )
    self.layer0 = nn.Sequential(*layers)
    ####
    layers = []
    layers.append( torch.nn.BatchNorm2d(nh) )
    if is_leaky:
      layers.append( torch.nn.LeakyReLU(inplace=True,negative_slope=0.2) )
    else:
      layers.append( torch.nn.ReLU(inplace=True) )
    layers.append( torch.nn.Conv2d(nh, nh, kernel_size=3, padding=1, stride=1, groups=ngroup) )
    self.layer1 = nn.Sequential(*layers)
    initialize_net(self)

  def forward(self, x):
    x = self.layer0(x)
    y = self.layer1(x)
    return torch.cat((y,x),1)

class ResUnit_BRC_ResHalf_Add_B(torch.nn.Module):
  def __init__(self, nc, is_leaky=True, is_separable=True):
    super(ResUnit_BRC_ResHalf_Add_B, self).__init__()
    ngroup = nc if is_separable else 1
    ####
    layers = []
    layers.append( torch.nn.BatchNorm2d(nc) )
    if is_leaky:
      layers.append( torch.nn.LeakyReLU(inplace=True,negative_slope=0.2) )
    else:
      layers.append( torch.nn.ReLU(inplace=True) )
    layers.append( torch.nn.Conv2d(nc, nc*2, kernel_size=4, padding=1, stride=2, groups=ngroup) )
    self.layer0 = nn.Sequential(*layers)
    initialize_net(self)

  def forward(self, x0):
    x1 = torch.nn.functional.avg_pool2d(x0,kernel_size=2,stride=2)
    y = self.layer0(x0)
    return y + torch.cat((x1,x1),1)


class ResUnit_BRC_ResHalf_Add_C(torch.nn.Module):
  def __init__(self, nc, is_separable=False, is_leaky=True):
    super(ResUnit_BRC_ResHalf_Add_C, self).__init__()
    ngroup = nc if is_separable else 1
    ####
    layers = []
    layers.append( torch.nn.BatchNorm2d(nc) )
    if is_leaky:
      layers.append( torch.nn.LeakyReLU(inplace=True,negative_slope=0.2) )
    else:
      layers.append( torch.nn.ReLU(inplace=True) )
    layers.append( torch.nn.Conv2d(nc, nc, kernel_size=4, padding=1, stride=2, groups=ngroup) )
    self.layer0 = nn.Sequential(*layers)
    initialize_net(self)

  def forward(self, x0):
    x1 = torch.nn.functional.avg_pool2d(x0,kernel_size=2,stride=2)
    y = self.layer0(x0)
    return y+x1

class ResUnit_BRC_ResDouble(torch.nn.Module):
  def __init__(self, nc_in,nc_out,is_separable=False):
    super(ResUnit_BRC_ResDouble, self).__init__()
    ngroup = nc_out if is_separable else 1
    self.bn1 = torch.nn.BatchNorm2d(nc_in)
    self.conv1 = torch.nn.ConvTranspose2d(nc_in, nc_out, kernel_size=1, padding=0, stride=1)
    ###
    self.model = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(nc_in, nc_out, kernel_size=2, padding=0, stride=2),
      torch.nn.BatchNorm2d(nc_out),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc_out, nc_out, kernel_size=3, padding=1, stride=1,groups=ngroup),
    )
    initialize_net(self)

  def forward(self, x):
    x = torch.nn.functional.relu(self.bn1(x))
    y = torch.nn.functional.interpolate(self.conv1(x),scale_factor=2)
    return y+self.model(x)


class ResUnit_BRC_ResDouble_Cat(torch.nn.Module):
  def __init__(self, nc_in,nc_out,is_separable=False):
    super(ResUnit_BRC_ResDouble_Cat, self).__init__()
    nh0 = nc_out//2
    nh1 = nc_out-nh0
    ngroup = nh0 if is_separable else 1
    self.bn1 = torch.nn.BatchNorm2d(nc_in)
    self.conv1 = torch.nn.ConvTranspose2d(nc_in, nh1, kernel_size=1, padding=0, stride=1)
    ###
    self.model = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(nc_in, nh0, kernel_size=2, padding=0, stride=2),
      torch.nn.BatchNorm2d(nh0),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nh0, nh0, kernel_size=3, padding=1, stride=1, groups=ngroup),
    )
    initialize_net(self)

  def forward(self, x):
    x = torch.nn.functional.relu(self.bn1(x))
    y = torch.nn.functional.interpolate(self.conv1(x),scale_factor=2)
    return torch.cat((y,self.model(x)),1)






