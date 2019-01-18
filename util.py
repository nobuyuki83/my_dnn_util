from OpenGL.GL import *
from OpenGL.GLUT import *
import cv2, math, hashlib, os, numpy, argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.filters import maximum_filter

def load_texture(img_bgr):
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  width_org = img_rgb.shape[1]
  height_org = img_rgb.shape[0]
  width_po2 = int(math.pow(2, math.ceil(math.log(width_org, 2))))
  height_po2 = int(math.pow(2, math.ceil(math.log(height_org, 2))))
  img_rgb = cv2.copyMakeBorder(img_rgb,
                               0, height_po2 - height_org, 0, width_po2 - width_org,
                               cv2.BORDER_CONSTANT, (0, 0, 0))
  rw = width_org / width_po2
  rh = height_org / height_po2
  glEnable(GL_TEXTURE_2D)
  texid = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, texid)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_po2, height_po2, 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
  return (width_org, height_org, rw, rh, texid)

def drawRect(rect, color=(1, 0, 0), width=1):
  plt = (rect[0], -rect[1])
  size_rect_w = rect[2]
  size_rect_h = rect[3]
  glDisable(GL_LIGHTING)
  glDisable(GL_TEXTURE_2D)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINE_LOOP)
  glVertex2f(plt[0], plt[1])
  glVertex2f(plt[0] + size_rect_w, plt[1])
  glVertex2f(plt[0] + size_rect_w, plt[1] - size_rect_h)
  glVertex2f(plt[0], plt[1] - size_rect_h)
  glEnd()

def drawCircle(cnt, rad, color=(1, 0, 0), width=1):
  glDisable(GL_TEXTURE_2D)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINE_LOOP)
  ndiv = 32
  dt = 3.1415*2.0/ndiv
  for i in range(32):
    glVertex3f(+cnt[0]+rad*math.cos(dt*i),
               -cnt[1]+rad*math.sin(dt*i), -0.1)
  glEnd()
  ###


def drawLine(cnt0, cnt1, color=(1, 0, 0), width=1):
  glDisable(GL_TEXTURE_2D)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINES)
  glVertex3f(+cnt0[0],-cnt0[1], -0.1)
  glVertex3f(+cnt1[0],-cnt1[1], -0.1)
  glEnd()


def drawPolyline(pl, color=(1, 0, 0), width=1):
  glDisable(GL_TEXTURE_2D)
  glDisable(GL_LIGHTING)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINE_LOOP)
  for ipl in range(len(pl)//2):
    glVertex2f(pl[ipl*2+0], -pl[ipl*2+1])
  glEnd()


def draw_keypoint_circle(dict_info,head_ratio,color,width,key):
  if not "face_rad" in dict_info:
    return
  r0 = dict_info["face_rad"]
  if key in dict_info:
    if not dict_info[key][2] == 0:
       drawCircle(dict_info[key],r0*head_ratio,color=color,width=width)

def draw_keypoint_line(dict_info,color,key0,key1):
  if key0 in dict_info and key1 in dict_info:
    if not dict_info[key0][2] == 0 and not dict_info[key1][2] ==0:
      drawLine(dict_info[key0],dict_info[key1],color,width=2)



def draw_annotation_keypoint(dict_info):
  draw_keypoint_circle(dict_info,1.0,(255,  0,  0),2,"keypoint_head")
  ####
  draw_keypoint_circle(dict_info,0.4,(  0,  0,255),2,"keypoint_shoulderleft")
  draw_keypoint_circle(dict_info,0.4,(  0,255,  0),2,"keypoint_shoulderright")
  draw_keypoint_circle(dict_info,0.3,(255,  0,255),2,"keypoint_elbowleft")
  draw_keypoint_circle(dict_info,0.3,(255,255,  0),2,"keypoint_elbowright")
  draw_keypoint_circle(dict_info,0.2,(  0,  0,255),2,"keypoint_wristleft")
  draw_keypoint_circle(dict_info,0.2,(  0,255,  0),2,"keypoint_wristright")
  ####
  draw_keypoint_circle(dict_info,0.7,(  0,  0,255),3,"keypoint_hipleft")
  draw_keypoint_circle(dict_info,0.7,(  0,255,  0),3,"keypoint_hipright")
  draw_keypoint_circle(dict_info,0.5,(255,  0,255),3,"keypoint_kneeleft")
  draw_keypoint_circle(dict_info,0.5,(255,255,  0),3,"keypoint_kneeright")
  draw_keypoint_circle(dict_info,0.3,(  0,  0,255),3,"keypoint_ankleleft")
  draw_keypoint_circle(dict_info,0.3,(  0,255,  0),3,"keypoint_ankleright")
  ####
  draw_keypoint_line(dict_info,(  0,  0,255), "keypoint_shoulderleft","keypoint_elbowleft")
  draw_keypoint_line(dict_info,(  0,255,  0), "keypoint_shoulderright","keypoint_elbowright")
  draw_keypoint_line(dict_info,(255,  0,255), "keypoint_elbowleft","keypoint_wristleft")
  draw_keypoint_line(dict_info,(255,255,  0), "keypoint_elbowright","keypoint_wristright")
  ####
  draw_keypoint_line(dict_info,(  0,  0,255), "keypoint_hipleft","keypoint_kneeleft")
  draw_keypoint_line(dict_info,(  0,255,  0), "keypoint_hipright","keypoint_kneeright")
  draw_keypoint_line(dict_info,(255,  0,255), "keypoint_kneeleft","keypoint_ankleleft")
  draw_keypoint_line(dict_info,(255,255,  0), "keypoint_kneeright","keypoint_ankleright")


def draw_annotation_bbox(dict_info):
  if 'bbox' in dict_info:
    drawRect(dict_info["bbox"],color=(255,0,0),width=1)


def draw_annotation_segmentation(dict_info,selected_loop:int):
  if 'segmentation' in dict_info:
    for iloop,loop in enumerate(dict_info['segmentation']):
      drawPolyline(loop,color=(1,1,1),width=1)
      if iloop == selected_loop:
        glColor3d(1.0,0.0,0.0)
      else:
        glColor3d(0.0,0.0,1.0)
      glPointSize(4)
      glBegin(GL_POINTS)
      for ip in range(len(loop)//2):
        x = loop[ip*2+0]
        y = loop[ip*2+1]
        glVertex2d(x,-y)
      glEnd()




def set_view_trans(img_size_info):
  viewport = glGetIntegerv(GL_VIEWPORT)
  win_h = viewport[3]
  win_w = viewport[2]
  img_w = img_size_info[0]
  img_h = img_size_info[1]
  #####
  scale_imgwin = max(img_h / win_h, img_w / win_w)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glOrtho(0, win_w * scale_imgwin, -win_h * scale_imgwin, 0, -1000, 1000)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

def draw_img(img_size_info):
  img_w = img_size_info[0]
  img_h = img_size_info[1]
  imgtex_w = img_size_info[2]
  imgtex_h = img_size_info[3]
  ####
  glDisable(GL_LIGHTING)
  glEnable(GL_TEXTURE_2D)
  id_tex_org = img_size_info[4]
  if id_tex_org is not None and glIsTexture(id_tex_org):
    glBindTexture(GL_TEXTURE_2D, id_tex_org)
  glColor3d(1, 1, 1)
  glBegin(GL_QUADS)
  ## left bottom
  glTexCoord2f(0.0, imgtex_h)
  glVertex2f(0, -img_h)
  ## right bottom
  glTexCoord2f(imgtex_w, imgtex_h)
  glVertex2f(img_w, -img_h)
  ### right top
  glTexCoord2f(imgtex_w, 0.0)
  glVertex2f(img_w, 0)
  ## left top
  glTexCoord2f(0.0, 0.0)
  glVertex2f(0, 0)
  glEnd()

def glut_print( x,  y,  font,  text, color):
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glColor3f(color[0],color[1],color[2])
  glRasterPos2f(x,y)
  for ch in text:
    glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )

def get_img_coord(xy, img_size_info):
  ####
  viewport = glGetIntegerv(GL_VIEWPORT)
  win_h = viewport[3]
  win_w = viewport[2]
  img_w = img_size_info[0]
  img_h = img_size_info[1]
#  print(win_h,win_w,img_h,img_w)
  ####
  scale_imgwin = max(img_h / win_h, img_w / win_w)
  x1 = xy[0] * scale_imgwin
  y1 = xy[1] * scale_imgwin
  return (x1, y1)

#####################################################################

def md5_hash(path):
  hash_md5 = hashlib.md5()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()

def is_dir(dirname):
  """Checks if a path is an actual directory"""
  if not os.path.isdir(dirname):
    msg = "{0} is not a directory".format(dirname)
    raise argparse.ArgumentTypeError(msg)
  else:
    return dirname

#############################################################


def cv2_draw_annotation(np_img0,dict_info,list_key,dict_key_prop):
  face_rad = dict_info["person0"]["face_rad"]
  for ikey, key in enumerate(list_key):
    if key in dict_info["person0"]:
      pos_key = dict_info["person0"][key]
      prop = dict_key_prop[key]
      color0 = prop[0:3]
      rad0 = prop[3]*face_rad
      width = int(prop[4])
      cv2.circle(np_img0, (pos_key[0], pos_key[1]), int(rad0), color0, width)

  if "bbox" in dict_info["person0"]:
    bbox = dict_info["person0"]["bbox"]
    cv2.rectangle(np_img0, tuple(bbox[0:2]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                (255, 0, 0), 1)

  if "segmentation" in dict_info["person0"]:
    list_seg = dict_info["person0"]["segmentation"]
    for seg in list_seg:
      np_sgm = numpy.array(seg,dtype=numpy.int)
      np_sgm = np_sgm.reshape((-1,2))
      cv2.polylines(np_img0, [np_sgm], True, (0, 255, 255))

def cv2_get_numpy_loop_array(list_loop):
  list_loop_np = []
  for loop in list_loop:
    list_loop_np.append(numpy.array(loop, dtype=int).reshape(-1, 2))
  return list_loop_np

def coco_get_image_annotation(path_json):
  dict_info = json.load(open(path_json))
  print(dict_info.keys())
  list_keypoint = (dict_info['categories'][0])['keypoints']
  print(list_keypoint)
  ####
  list_images = dict_info["images"]
  dict_imgid2iimg= {}
  for iimg,img in enumerate(list_images):
    id0 = int(img['id'])
    dict_imgid2iimg[id0] = iimg
  ####
  dict_iimg2ianno = {}
  for ianno,anno in enumerate(dict_info["annotations"]):
    imgid0 = int(anno["image_id"])
    iimg0 = dict_imgid2iimg[imgid0]
    if not iimg0 in dict_iimg2ianno:
      dict_iimg2ianno[iimg0] = [ianno]
    else:
      dict_iimg2ianno[iimg0].append(ianno)

  for iimg,img in enumerate(list_images):
    if not iimg in dict_iimg2ianno:
      continue
    if not len(dict_iimg2ianno[iimg]) == 1:
      continue

  list_imginfo_anno = []
  for iimg, img in enumerate(list_images):
    if not iimg in dict_iimg2ianno:
      continue
    if not len(dict_iimg2ianno[iimg]) == 1:
      continue
    ####
    ianno = dict_iimg2ianno[iimg][0] # only one annotation for this image
    anno = dict_info['annotations'][ianno]
    list_imginfo_anno.append([img,anno])
  return list_imginfo_anno


def coco_get_dict_info(imginfo,anno,list_keypoint_name):
#  file_name = imginfo['file_name']
  keypoints = list(map(lambda x: int(x), anno['keypoints']))
  np_sgm = []
  list_sgm = []
  for iseg, seg in enumerate(anno['segmentation']):
    seg0 = numpy.array(seg, numpy.int32).reshape((-1, 2))
    np_sgm.append(seg0)
    list_sgm.append(list(map(lambda x: int(x), seg)))
  bbox = list(map(lambda x: int(x), anno['bbox']))
  #print(imginfo)
#  print(anno)
#  print(bbox)
  ####
  dict_out = {}
  dict_out['coco_id'] = imginfo['id']
  dict_out["person0"] = {}
  dict_out["person0"]['face_rad'] = 40
  dict_out["person0"]['bbox'] = bbox
  dict_out["person0"]['segmentation'] = list_sgm
  assert len(keypoints) % 3 == 0
  for ip in range(len(keypoints) // 3):
    if keypoints[ip * 3 + 2] == 0:
      continue
    x0 = keypoints[ip * 3 + 0]
    y0 = keypoints[ip * 3 + 1]
    dict_out['person0'][list_keypoint_name[ip]] = [x0, y0, keypoints[ip * 3 + 2]]
  return dict_out


def arrange_old_new(list_path_img):
  dict_bn_time = {}
  for path_img in list_path_img:
    path_json = path_img.rsplit(".",1)[0]+".json"
    if not os.path.isfile(path_json):
      dict_bn_time[path_img] = -time.time()
      continue
    with open(path_json,"r") as file0:
      dict_info0 = json.load(file0)
      if "saved_time" in dict_info0:
        dict_bn_time[path_img] = -dict_info0["saved_time"]
      else:
        dict_bn_time[path_img] = -time.time()
  list0 = sorted(dict_bn_time.items(), key=lambda x:x[1])
  return list(map(lambda x:x[0],list0))

############################################################

def get_image_npix(img1, npix, mag):
  img2 = img1[0::mag, 0::mag]
  ####
  img3 = img2.copy()
  h3 = img3.shape[0]
  w3 = img3.shape[1]
  nbh3 = math.ceil(h3 / npix)
  nbw3 = math.ceil(w3 / npix)
  img3 = cv2.copyMakeBorder(img3, 0, npix * nbh3 - h3, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  img3 = cv2.copyMakeBorder(img3, 0, 0, 0, npix * nbw3 - w3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  return img3

def img_pad(img1, size):
  size1 = max(img1.shape[0],img1.shape[1])
  mag = size/size1
  img2 = cv2.resize(img1,(int(mag*img1.shape[1]),int(mag*img1.shape[0])))
  img3 = img2.copy()
  ####
  h3 = img3.shape[0]
  w3 = img3.shape[1]
  img3 = cv2.copyMakeBorder(img3, 0, size - h3, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  img3 = cv2.copyMakeBorder(img3, 0, 0, 0, size - w3, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  return img3, 1.0/mag



def img_pad_mag_center(img1, size, mag, cnt):
  img2 = cv2.resize(img1,(int(mag*img1.shape[1]),int(mag*img1.shape[0])))
  cnt2 = [cnt[0]*mag,cnt[1]*mag]
  ####
  rl = int(cnt2[0]-size[0]//2)
  rr = rl+size[0]
  rt = int(cnt2[1]-size[1]//2)
  rb = rt+size[1]
  bl = max(-rl,0)
  br = max(rr-img2.shape[1],0)
  bt = max(-rt,0)
  bb = max(rb-img2.shape[0],0)
  img2 = cv2.copyMakeBorder(img2, bt, bb, bl, br, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  ####
  cx0 = rl+bl
  cy0 = rt+bt
  ####
  img3 = img2[cy0:cy0+size[1],cx0:cx0+size[0],:]
  return img3



def gauss_keypoint(np0, idim, cx0,cy0,r0):
  nh = np0.shape[0]
  nw = np0.shape[1]
  w0 = numpy.arange(0, nw, 1, numpy.float32)
  h0 = numpy.arange(0, nh, 1, numpy.float32)
  w0 = w0.reshape((1,nw))
  h0 = h0.reshape((nh,1))
  np0[:,:,idim] = numpy.exp(-((w0-cx0)**2+(h0-cy0)**2)/(r0*r0))


def view_batch(np_in, np_tg, nstride):
  '''
  np_in = [nbatch,nh*npix,nw*npix,3]
  np_tg = [nbatch,nh,nw,nch]
  '''
  assert np_in.shape[0] == np_tg.shape[0]
  assert np_in.shape[1] == np_tg.shape[1]*nstride
  assert np_in.shape[2] == np_tg.shape[2]*nstride
  assert np_in.shape[3] == 3
  assert np_in.dtype == numpy.uint8
  assert np_tg.max() <= 1.1 and np_tg.min() >= -0.1
  assert np_tg.dtype == numpy.float32
  #####
  nch_tg = np_tg.shape[3]
  for ibatch in range(np_in.shape[0]):
    for itr in range((nch_tg+2)//3):
      np_tg1 = numpy.zeros((np_tg.shape[1],np_tg.shape[2],3),dtype=numpy.float32)
      if itr*3+0 < nch_tg: np_tg1[:,:,0] = np_tg[ibatch,:,:,itr*3+0]
      if itr*3+1 < nch_tg: np_tg1[:,:,1] = np_tg[ibatch,:,:,itr*3+1]
      if itr*3+2 < nch_tg: np_tg1[:,:,2] = np_tg[ibatch,:,:,itr*3+2]
      np_tg1 = ((cv2.resize(np_tg1, None, fx=nstride, fy=nstride)) * 255.0).astype(numpy.uint8)
      np_tg1 = cv2.cvtColor(np_tg1,cv2.COLOR_RGB2BGR)
      added_image = cv2.addWeighted(np_tg1, 0.9, np_in[ibatch], 0.3, 2.0)
      cv2.imshow("hoge", added_image.astype(numpy.uint8))
      cv2.waitKey(-1)


##################################################################################################

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


def get_peaks(list_key,np_out0,mag):
  pos_key = [[0, 0, 0]] * len(list_key)
  for ikey, key in enumerate(list_key):
    local_max = maximum_filter(np_out0[:, :, ikey], footprint=numpy.ones((5, 5))) == np_out0[:, :, ikey]
    local_max = local_max * (np_out0[:, :, ikey] > 0.2)
    peaks = ((numpy.array(numpy.nonzero(local_max)[::-1]).T) * (2.0 / mag)).astype(numpy.int)
    if peaks.shape[0] == 1:
      pos_key[ikey] = [peaks[0][0], peaks[0][1], 2]
  return pos_key


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






