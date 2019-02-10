import os, cv2, math
import torch
import torch.nn.functional
import my_dnn_util.util_torch as my_torch
import my_dnn_util.util as my_util

class NetDetect_s32(torch.nn.Module):
  def __init__(self,path_file):
    super(NetDetect_s32, self).__init__()
    self.nstride = 32
    self.npix = 32
    self.path_file = path_file
    '''
    self.layer_dsc = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=1),
      my_torch.ModuleBRC_Half_k4s2(64,128), # 1/2
      my_torch.ModuleBRC_Half_k4s2(128,256), # 1/4
      my_torch.ModuleBRC_Half_k4s2(256,256), # 1/8
      my_torch.ModuleBRC_Half_k4s2(256,256), # 1/16
      my_torch.ModuleBRC_Half_k4s2(256,256)  # 1/32
    )
    '''
    self.layer_dsc = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=1),
      my_torch.ModuleBRC_Half_ResAdd(64,128,nc_in_group=2), # 1/2
      my_torch.ModuleBRC_ResBtl(128,nc_in_group=4),
      my_torch.ModuleBRC_Half_ResAdd(128,256,nc_in_group=2), # 1/4
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      my_torch.ModuleBRC_Half_ResAdd(256,256,nc_in_group=2), # 1/8
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      my_torch.ModuleBRC_Half_ResAdd(256,256,nc_in_group=2), # 1/16
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      my_torch.ModuleBRC_Half_ResAdd(256,256,nc_in_group=2),  # 1/32
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
    )
    '''
    self.layer_dsc = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=1),
      torch.nn.MaxPool2d(kernel_size=2,stride=2), # 1/2
      my_torch.ModuleBRC_ResAdd(64),
      my_torch.ModuleBRC_Half_k4s2(64, 128),  # 1/4
      my_torch.ModuleBRC_ResAdd(128),
      my_torch.ModuleBRC_Half_k4s2(128, 256),  # 1/8
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_Half_k4s2(256, 256),  # 1/16
      my_torch.ModuleBRC_ResAdd(256),
      my_torch.ModuleBRC_Half_k4s2(256, 256),  # 1/32
    )
    '''
    self.layer_cls = torch.nn.Sequential(
      my_torch.ModuleBRC_k1(256,64),
      my_torch.ModuleBRC_k1(64,32),
      my_torch.ModuleBRC_k1(32,2),
    )
    self.layer_prm = torch.nn.Sequential(
      my_torch.ModuleBRC_k1(256,64,af='tanh'),
      my_torch.ModuleBRC_k1(64, 32,af='tanh'),
      my_torch.ModuleBRC_k1(32,  3,af='tanh'),
      torch.nn.Tanh()
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(self.path_file):
      self = my_torch.load_model_cpm(self, self.path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def dsc(self, x):
    return self.layer_dsc(x)

  def cls(self,x):
    return self.layer_cls(x)

  def prm(self,x):
    return self.layer_prm(x)

  def forward(self,vpt_in):
    out_dsc = self.dsc(vpt_in)
    out_cls = self.cls(out_dsc)
    out_prm = self.prm(out_dsc)
    return out_cls, out_prm


class NetDetect_s16_A(torch.nn.Module):
  def __init__(self,path_file):
    super(NetDetect_s16_A, self).__init__()
    self.nstride = 16
    self.npix = 16
    self.path_file = path_file
    #####
    self.layer_dsc = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=1),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/2
      my_torch.ModuleBRC_k1( 64,128),
      my_torch.ModuleBRC_ResBtl(128,nc_in_group=4),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/4
      my_torch.ModuleBRC_k1(128, 256),
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/8
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/16
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
    )
    self.layer_cls = torch.nn.Sequential(
      my_torch.ModuleBRC_k1(256,64),
      my_torch.ModuleBRC_k1(64,2),
    )
    self.layer_prm = torch.nn.Sequential(
      my_torch.ModuleBRC_k1(256,64,af='tanh'),
      my_torch.ModuleBRC_k1(64,  3,af='tanh'),
      torch.nn.Tanh()
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(self.path_file):
      self = my_torch.load_model_cpm(self, self.path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def dsc(self, x):
    return self.layer_dsc(x)

  def cls(self,x):
    return self.layer_cls(x)

  def prm(self,x):
    return self.layer_prm(x)

  def forward(self,vpt_in):
    out_dsc = self.dsc(vpt_in)
    out_cls = self.cls(out_dsc)
    out_prm = self.prm(out_dsc)
    return out_cls, out_prm


class NetDetect_s16_B(torch.nn.Module):
  def __init__(self,path_file):
    super(NetDetect_s16_B, self).__init__()
    self.nstride = 16
    self.npix = 16
    self.path_file = path_file
    #####
    self.layer_dsc = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=1),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/2
      my_torch.ModuleBRC_k1( 64,128),
      my_torch.ModuleBRC_ResBtl(128,nc_in_group=4),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/4
      my_torch.ModuleBRC_k1(128, 256),
      my_torch.ModuleBRC_ResBtl(256,nc_in_group=4),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/8
      my_torch.ModuleBRC_k1(256, 512),
      my_torch.ModuleBRC_ResBtl(512,nc_in_group=4),
      my_torch.ModuleBRC_ResBtl(512,nc_in_group=4),
      torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1/16
      my_torch.ModuleBRC_ResBtl(512,nc_in_group=4),
      my_torch.ModuleBRC_ResBtl(512,nc_in_group=4),
    )
    self.layer_cls = torch.nn.Sequential(
      my_torch.ModuleBRC_k1(512,64),
      my_torch.ModuleBRC_k1(64,2),
    )
    self.layer_prm = torch.nn.Sequential(
      my_torch.ModuleBRC_k1(512,64,af='tanh'),
      my_torch.ModuleBRC_k1(64,  3,af='tanh'),
      torch.nn.Tanh()
    )
    my_torch.initialize_net(self)
    ####
    if os.path.isfile(self.path_file):
      self = my_torch.load_model_cpm(self, self.path_file)
    if torch.cuda.is_available():
      self = self.cuda()

  def dsc(self, x): return self.layer_dsc(x)
  def cls(self,x): return self.layer_cls(x)
  def prm(self,x): return self.layer_prm(x)

  def forward(self,vpt_in):
    out_dsc = self.dsc(vpt_in)
    out_cls = self.cls(out_dsc)
    out_prm = self.prm(out_dsc)
    return out_cls, out_prm


##################################

def detection_view_out(np_out,np_cls,np_prm,nstride):
  nblkh = np_out.shape[0] // nstride
  nblkw = np_out.shape[1] // nstride
  for iblkh in range(nblkh):
    for iblkw in range(nblkw):
      prob0 = np_cls[0][iblkh][iblkw]
      prob1 = np_cls[1][iblkh][iblkw]
      prob1a = math.exp(prob1) / (math.exp(prob0) + math.exp(prob1))
      if prob1a < 0.8: continue
      my_util.cv2_draw_circle(np_out, (iblkw * nstride + nstride // 2, iblkh * nstride + nstride // 2),
                              rad=nstride*0.72,
                              color=(0, 255, 0))
      p0x = iblkw * nstride + nstride // 2
      p0y = iblkh * nstride + nstride // 2
      p1x = np_prm[0][iblkh][iblkw] * 2.0 * nstride + p0x
      p1y = np_prm[1][iblkh][iblkw] * 2.0 * nstride + p0y
      rs = math.pow(2, np_prm[2][iblkh][iblkw] * 2.0) * nstride
      my_util.cv2_draw_circle(np_out, (p1x, p1y), rs, color=(0, 0, 255))
  cv2.imshow("hoge", np_out)
  cv2.waitKey(-1)


def detect(img_in,net_detect,mag:int,prob_thre):
  nstride = net_detect.nstride
  img_bgr = img_in[::mag,::mag,:]
  np_bgr = my_util.get_image_npix(img_bgr, npix=nstride)
  vpt_in = my_torch.np2pt_img(np_bgr, scale=2.0 / 255.0, offset=-1.0, requires_grad=False)
  with torch.no_grad():
    out_cls, out_prm = net_detect(vpt_in)
  np_cls = out_cls.data.numpy()[0]
  np_prm = out_prm.data.numpy()[0]
  ###
  nblkh = np_bgr.shape[0] // nstride
  nblkw = np_bgr.shape[1] // nstride
  ###
  list_out = []
  for iblkh in range(nblkh):
    for iblkw in range(nblkw):
      prob0 = np_cls[0][iblkh][iblkw]
      prob1 = np_cls[1][iblkh][iblkw]
      #          prob0a = math.exp(prob0)/(math.exp(prob0)+math.exp(prob1))
      prob1a = math.exp(prob1) / (math.exp(prob0) + math.exp(prob1))
      if prob1a < prob_thre:
        continue
      p1x = np_prm[0][iblkh][iblkw] * 2.0 * nstride + iblkw * nstride + nstride // 2
      p1y = np_prm[1][iblkh][iblkw] * 2.0 * nstride + iblkh * nstride + nstride // 2
      rs = math.pow(2, np_prm[2][iblkh][iblkw] * 2.0) * nstride
      list_out.append([p1x*mag,p1y*mag,rs*mag,mag,prob1a])
  return list_out


def detect_multires(img_bgr,net_detect):
  list_out = [] # px,py,rad,mag,prob1a
  ####
  len_img = (img_bgr.shape[0]*img_bgr.shape[1])/(net_detect.nstride*net_detect.nstride)
#  print(len_img)
  prob_thre = 0.7
  size_thre = 1500
  if 10 < len_img/16 < size_thre:
#    print("four")
    list_out.extend( detect(img_bgr,net_detect,mag=4,prob_thre=prob_thre) )
  if 10 < len_img/4 < size_thre and len(list_out)<2:
#    print("half")
    list_out.extend( detect(img_bgr,net_detect,mag=2,prob_thre=prob_thre) )
  if 10 < len_img/1 < size_thre and len(list_out)<2:
#    print("one")
    list_out.extend( detect(img_bgr,net_detect,mag=1,prob_thre=prob_thre) )
  ####
  list_group = []
  for ic in range(len(list_out)):
    igroup_in = -1
    for jgroup,list_ic_in_group in enumerate(list_group):
      for jc in list_ic_in_group:
        dx = list_out[ic][0]-list_out[jc][0]
        dy = list_out[ic][1]-list_out[jc][1]
        if math.sqrt(dx**2+dy**2) < list_out[ic][2]+list_out[jc][2]:
          igroup_in = jgroup
          break
      if igroup_in != -1: break
    if igroup_in == -1:
      list_group.append([ic])
    else:
      list_group[igroup_in].append(ic)
  list_group.sort(key=lambda x: -len(x))
#  print(list_group)
  ####
  if len(list_group) == 0:
    return None

  px,py,rd = 0,0,0
  for ic in list_group[0]:
    px += list_out[ic][0]
    py += list_out[ic][1]
    rd += list_out[ic][2]
  px /= len(list_group[0])
  py /= len(list_group[0])
  rd /= len(list_group[0])
  return [px,py,rd]