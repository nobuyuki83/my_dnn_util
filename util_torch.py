import cv2, math, hashlib, os, numpy, argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_segmentation_map(net_seg, np_img, mag):
  npix = net_seg.npix
  net_seg.eval()
  np_in = cv2.resize(np_img, (int(mag * np_img.shape[1]), int(mag * np_img.shape[0])))
  np_in = get_image_npix(np_in, npix, 1)
  np_in = np_in.reshape([1] + list(np_in.shape))
  ####
  pt_in = torch.from_numpy(numpy.moveaxis(np_in, 3, 1).astype(numpy.float32) / 255.0)
  with torch.no_grad():
    pt_out0 = net_seg.forward(pt_in)
  np_out0 = numpy.moveaxis(pt_out0.data.numpy(), 1, 3)
#    view_batch(np_in,np_out0,self.net_cpm0.nstride)
  ####
  np_out0 = np_out0.reshape(np_out0.shape[1:])
  np_in = np_in.reshape(np_in.shape[1:])
  return np_in,np_out0


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
    if isinstance(m, torch.nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()


class ResUnit_BRC_Btl(torch.nn.Module):
  def __init__(self, nc):
    super(ResUnit_BRC_Btl, self).__init__()
    assert nc%2 == 0
    nh = nc//2
    self.net = torch.nn.Sequential(
      torch.nn.BatchNorm2d(nc),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc, nh, kernel_size=1),
      torch.nn.BatchNorm2d(nh),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nh, nh, kernel_size=3, padding=1),
      torch.nn.BatchNorm2d(nh),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nh, nc, kernel_size=1),
    )
    initialize_net(self)

  def forward(self, x):
    return self.net(x)+x



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
  def __init__(self, nc_in, nc_out):
    super(ResUnit_BRC_ResHalf, self).__init__()
    self.bn1 = torch.nn.BatchNorm2d(nc_in)
    self.conv1 = torch.nn.Conv2d(nc_in, nc_out, kernel_size=1, padding=0, stride=1)
    ###
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(nc_in, nc_out, kernel_size=4, padding=1, stride=2),
      torch.nn.BatchNorm2d(nc_out),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc_out, nc_out, kernel_size=3, padding=1,stride=1),
    )
    initialize_net(self)

  def forward(self, x):
    x = torch.nn.functional.relu(self.bn1(x))
    y = torch.nn.functional.max_pool2d(self.conv1(x),kernel_size=4,padding=1,stride=2)
    return y+self.model(x)


class ResUnit_BRC_ResDouble(torch.nn.Module):
  def __init__(self, nc_in,nc_out):
    super(ResUnit_BRC_ResDouble, self).__init__()
    self.bn1 = torch.nn.BatchNorm2d(nc_in)
    self.conv1 = torch.nn.ConvTranspose2d(nc_in, nc_out, kernel_size=1, padding=0, stride=1)
    ###
    self.model = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(nc_in, nc_out, kernel_size=2, padding=0, stride=2),
      torch.nn.BatchNorm2d(nc_out),
      torch.nn.ReLU(inplace=True),
      torch.nn.Conv2d(nc_out, nc_out, kernel_size=3, padding=1, stride=1),
    )
    initialize_net(self)

  def forward(self, x):
    x = torch.nn.functional.relu(self.bn1(x))
    y = torch.nn.functional.interpolate(self.conv1(x),scale_factor=2)
    return y+self.model(x)






