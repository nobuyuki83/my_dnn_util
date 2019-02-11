import os, cv2, math, numpy, json, glob, random
import torch
import torch.nn.functional
import my_dnn_util.util_torch as my_torch
import my_dnn_util.util as my_util
import my_dnn_util.dsp.util as my_dsp

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

####################################

def affine_random_head(dict_info,size_input, size_output, target_rad):
  dict_prsn = dict_info["person0"]
  scale = target_rad / dict_prsn["rad_head"]
  scale *= random.uniform(0.5, 2.0)
  if 'kp_head' in dict_prsn:
    x0 = dict_prsn["kp_head"][0]
    y0 = dict_prsn["kp_head"][1]
    cnt = [x0+random.randint(-64,64), y0+random.randint(-64,64)] # x,y
  else:
    cnt = [size_input[0] * 0.5, + size_input[1] * 0.5]
  rot = random.randint(-40, 40)
  rot_mat = cv2.getRotationMatrix2D(tuple([cnt[0], cnt[1]]), rot, scale)
  rot_mat[0][2] += size_output[0] * 0.5 - cnt[0] + random.randint(int(-8 / scale), int(+8 / scale))
  rot_mat[1][2] += size_output[1] * 0.5 - cnt[1] + random.randint(int(-8 / scale), int(+8 / scale))
  return rot_mat, scale

class BatchesHead:
  def __init__(self, list_path_dir_img:list, nbatch):
    list_path_json = []
    for path_dir_img in list_path_dir_img:
      list_path_json += glob.glob(path_dir_img + "/*.json")
    self.bd = my_util.BatchDispenser(nbatch, list_path_json)

  def get_batch(self,nstride):
    size_img = 32*8
    nblk = size_img // nstride
    np_batch_in = numpy.empty((0, size_img, size_img, 3), dtype=numpy.uint8)
    np_batch_tgc = numpy.empty((0, nblk, nblk), dtype=numpy.int64)
    np_batch_tgl = numpy.empty((0, 3, nblk, nblk), dtype=numpy.float32)
    np_batch_msk = numpy.empty((0, nblk, nblk), dtype=numpy.float32)
    ####
    for path_json in self.bd.get_batch_path():
      dict_info = json.load(open(path_json, "r"))
      np_img0 = cv2.imread(my_util.path_img_same_name(path_json))
      rot_mat, scale = affine_random_head(dict_info,
                                          (np_img0.shape[1], np_img0.shape[0]), (size_img, size_img),
                                          target_rad=nstride*0.72)
      np_img1 = cv2.warpAffine(np_img0, rot_mat, (size_img, size_img), flags=cv2.INTER_CUBIC)
      np_tgc, np_tgl, np_msk = my_dsp.input_detect("kp_head",dict_info,nblk,nstride,rot_mat,scale)
      ####
      np_batch_in = numpy.vstack((np_batch_in, np_img1.reshape(1, size_img, size_img, 3)))
      np_batch_tgc = numpy.vstack((np_batch_tgc, np_tgc))
      np_batch_tgl = numpy.vstack((np_batch_tgl, np_tgl))
      np_batch_msk = numpy.vstack((np_batch_msk, np_msk))
    np_batch_in = np_batch_in.astype(numpy.uint8)
    return np_batch_in,np_batch_tgc,np_batch_tgl,np_batch_msk

  def get_batch_vpt(self,nstride):
    np_in, np_tgc, np_tgl, np_msk = self.get_batch(nstride)
    vpt_in = my_torch.np2pt_img(np_in,scale=2.0/255.0,offset=-1.0,requires_grad=True)
    vpt_tgc = torch.autograd.Variable(torch.from_numpy(np_tgc), requires_grad=False)
    vpt_tgl = torch.autograd.Variable(torch.from_numpy(np_tgl), requires_grad=False)
    vpt_msk = torch.autograd.Variable(torch.from_numpy(np_msk), requires_grad=False)
    if torch.cuda.is_available():
      vpt_tgc = vpt_tgc.cuda()
      vpt_tgl = vpt_tgl.cuda()
      vpt_msk = vpt_msk.cuda()
    return vpt_in,vpt_tgc,vpt_tgl,vpt_msk


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

def view_in_tg(np_in,np_tgc,np_tgl,np_msk,nstride):
  nblkh = np_in.shape[0]//nstride
  nblkw = np_in.shape[1]//nstride
  assert np_tgc.shape == (nblkh,nblkw)
  assert np_tgl.shape == (3,nblkh,nblkw)
  np_out = np_in.copy()
  for iblkh in range(nblkh):
    for iblkw in range(nblkw):
      if np_msk[iblkh][iblkw] == 1:
        my_util.cv2_draw_circle(np_out,
                                ((iblkw + 0.5) * nstride, (iblkh + 0.5) * nstride),
                                color=(0, 255, 0), rad=nstride*0.72)
      if np_tgc[iblkh][iblkw] != 1:
        continue
      p0x = iblkw*nstride+nstride//2
      p0y = iblkh*nstride+nstride//2
      print(np_tgl[0][iblkh][iblkw],np_tgl[1][iblkh][iblkw],np_tgl[2][iblkh][iblkw])
      p1x = np_tgl[0][iblkh][iblkw]*2.0*nstride + p0x
      p1y = np_tgl[1][iblkh][iblkw]*2.0*nstride + p0y
      rs = math.pow(2,np_tgl[2][iblkh][iblkw]*2.0)*nstride
      my_util.cv2_draw_circle(np_out,(p1x,p1y),rs,color=(0,0,255))
      my_util.cv2_draw_line(np_out,(p1x,p1y),(p0x,p0y),color=(0,255,0))
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
  net_detect.eval()
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
    return {}

  px,py,rd = 0,0,0
  for ic in list_group[0]:
    px += list_out[ic][0]
    py += list_out[ic][1]
    rd += list_out[ic][2]
  px /= len(list_group[0])
  py /= len(list_group[0])
  rd /= len(list_group[0])
  ####
  dict_info0 = {}
  dict_info0["person0"] = {}
  dict_info0["person0"]["rad_head"] = rd
  dict_info0["person0"]["kp_head"] = [px, py, 2]
  return dict_info0