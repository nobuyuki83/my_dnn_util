import os
import torch
import torch.nn.functional
import torch.utils.model_zoo as model_zoo
from torchvision import transforms

import my_dnn_util.util_torch as my_torch


##################################################################################################


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
