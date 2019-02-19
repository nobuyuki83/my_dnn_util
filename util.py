import cv2, math, hashlib, os, numpy, argparse, json, time, random
#import torch
from scipy.ndimage.filters import maximum_filter

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

def path_img_same_name(path_json:str):
  if os.path.isfile(path_json.rsplit(".", 1)[0] + ".jpg"):
    path_img = path_json.rsplit(".", 1)[0] + ".jpg"
  else:
    path_img = path_json.rsplit(".", 1)[0] + ".png"
  return path_img

##################################
# numpy

def gauss_keypoint(np0, idim, cx0,cy0,r0):
  nh = np0.shape[0]
  nw = np0.shape[1]
  w0 = numpy.arange(0, nw, 1, numpy.float32)
  h0 = numpy.arange(0, nh, 1, numpy.float32)
  w0 = w0.reshape((1,nw))
  h0 = h0.reshape((nh,1))
  np0[:,:,idim] = numpy.exp(-((w0-cx0)**2+(h0-cy0)**2)/(r0*r0))


def pca_color_augmentation(np_bgr_in):
  assert np_bgr_in.ndim == 3 and np_bgr_in.shape[2] == 3
  assert np_bgr_in.dtype == numpy.uint8

  img = np_bgr_in.reshape(-1, 3).astype(numpy.float32)
  img = (img - numpy.mean(img, axis=0)) / numpy.std(img, axis=0)

  cov = numpy.cov(img, rowvar=False)
  lambd_eigen_value, p_eigen_vector = numpy.linalg.eig(cov)

  rand = numpy.random.randn(3) * 0.1
  delta = numpy.dot(p_eigen_vector, rand * lambd_eigen_value)
  delta = (delta * 255.0).astype(numpy.int32)[numpy.newaxis, numpy.newaxis, :]

  np_bgr_out = numpy.clip(np_bgr_in + delta, 0, 255).astype(numpy.uint8)
  return np_bgr_out


def get_random_polygon(x0,y0,rad):
  ndiv = random.randint(3,6)
  delrad = math.pi*2.0/ndiv
  list_loop = []
  for idiv in range(ndiv):
    r1 = rad*random.uniform(1.0,2.0)
    x1 = r1*math.cos(delrad*(idiv+random.uniform(0.2,0.8)))
    y1 = r1*math.sin(delrad*(idiv+random.uniform(0.2,0.8)))
    list_loop.append(x0+x1)
    list_loop.append(y0+y1)
  return list_loop



class BatchDispenser:
  def __init__(self, nbatch:int, list_path_in:list):
    self.nbatch = nbatch
    self.ibatch = 0
    self.iepoch = 0
    self.list_path_in = list_path_in
    random.shuffle(self.list_path_in)

  def get_batch_path(self):
    list_path_batch_in = []
    for iipath in range(self.nbatch):
      ipath0 = self.ibatch*self.nbatch+iipath
      if ipath0 >= len(self.list_path_in):
        break
      list_path_batch_in.append(self.list_path_in[ipath0])
    self.ibatch += 1
    if self.ibatch*self.nbatch >= len(self.list_path_in):
      random.shuffle(self.list_path_in)
      self.ibatch = 0
      self.iepoch += 1
    return list_path_batch_in

#################################



class FaceDetectorCV:
  def __init__(self):
    path_data = cv2.data.haarcascades
    self.face_cascade = cv2.CascadeClassifier(path_data + 'haarcascade_frontalface_default.xml')

  def get_face(self, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = self.face_cascade.detectMultiScale(gray, minNeighbors=10)
    return faces


def cv2_get_numpy_loop_array(list_loop):
  list_loop_np = []
  for loop in list_loop:
    list_loop_np.append(numpy.array(loop, dtype=int).reshape(-1, 2))
  return list_loop_np



def cv2_draw_rect(img, rect:list, color:tuple, width=2):
  assert len(rect) == 4
  recti = list(map(int, rect))
  cv2.rectangle(img, (recti[0], recti[1]), (recti[0] + recti[2], recti[1] + recti[3]), color, width)

def cv2_draw_circle(np_img,cnt,rad,color:tuple,width=1):
  cv2.circle(np_img, (int(cnt[0]), int(cnt[1])), int(rad), color=color, thickness=width)

def cv2_draw_line(np_img,p0,p1,color:tuple):
  cv2.line(np_img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color=color)


################################################################################################

def intersect_rects(rec0, rec1):
  assert len(rec0) == 4
  assert len(rec1) == 4
  tlx = max(rec0[0], rec1[0])
  tly = max(rec0[1], rec1[1])
  brx = min(rec0[0] + rec0[2], rec1[0] + rec1[2])
  bry = min(rec0[1] + rec0[3], rec1[1] + rec1[3])
  if tlx > brx: return []
  if tly > bry: return []
  return [tlx, tly, brx - tlx, bry - tly]

def area_rect(rec0):
  if len(rec0) != 4: return 0.0
  return rec0[2] * rec0[3]

def iou_rect(trim0, trim2):
  trim3 = intersect_rects(trim0, trim2)
  an = area_rect(trim3)
  au = area_rect(trim2) + area_rect(trim0) - an
  iou = float(an) / float(au)
  return iou

def iou_square(dx,dy,w0,w1):
  rec0 = [-w0/2,-w0/2,w0,w0]
  rec1 = [dx-w1/2,dy-w1/2,w1,w1]
  return iou_rect(rec0,rec1)

def iou_circle(dx,dy,r0,r1):
  d = math.sqrt(dx**2+dy**2)
  if d > r0+r1: return 0.0
  r0s, r1s, ds = r0**2, r1**2, d**2
  a0 = r0s*math.pi
  a1 = r1s*math.pi
  if r0 > d+r1: return a1/a0
  if r1 > d+r0: return a0/a1
  alpha = math.acos((ds + r0s - r1s) / (2 * d * r0))
  beta = math.acos((ds + r1s - r0s) / (2 * d * r1))
  a01i = r0s * alpha + r1s * beta - 0.5 * (r0s * math.sin(2 * alpha) + r1s * math.sin(2 * beta))
  a01u = a0+a1-a01i
  return a01i/a01u

############################################################

def get_image_npix(img1, npix, mag=1, color=(0,0,0)):
  img2 = img1[0::mag, 0::mag]
  ####
  img3 = img2.copy()
  h3 = img3.shape[0]
  w3 = img3.shape[1]
  nbh3 = math.ceil(h3 / npix)
  nbw3 = math.ceil(w3 / npix)
  img3 = cv2.copyMakeBorder(img3, 0, npix * nbh3 - h3, 0, 0, cv2.BORDER_CONSTANT, value=color)
  img3 = cv2.copyMakeBorder(img3, 0, 0, 0, npix * nbw3 - w3, cv2.BORDER_CONSTANT, value=color)
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


def view_batch(np_in, np_tg, nstride, ratio=0.5):
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
      added_image = cv2.addWeighted(np_tg1, ratio, np_in[ibatch], (1-ratio), 1.0)
      cv2.imshow("hoge", added_image.astype(numpy.uint8))
      cv2.waitKey(-1)


##################################################################################################

'''
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
  return np_in,np_out0
'''

def get_peaks(list_key,np_out0,mag,thres=0.3):
  pos_key = [[0, 0, 0]] * len(list_key)
  assert np_out0.shape[2] == len(list_key)
  for ikey, key in enumerate(list_key):
    local_max = maximum_filter(np_out0[:, :, ikey], footprint=numpy.ones((5, 5))) == np_out0[:, :, ikey]
    local_max = local_max * (np_out0[:, :, ikey] > thres)
    peaks = numpy.array(numpy.nonzero(local_max)[::-1]).T
    list_peaks = []
    for ipeak in range(peaks.shape[0]):
      iw0 = peaks[ipeak][0]
      ih0 = peaks[ipeak][1]
      conf = np_out0[ih0,iw0,ikey]
      list_peaks.append([conf,int(iw0*2.0/mag),int(ih0*2.0/mag)])
    list_peaks.sort(key=lambda x:-x[0])
    if len(list_peaks) > 0:
      pos_key[ikey] = [list_peaks[0][1], list_peaks[0][2], 2]
  return pos_key


'''
def get_segmentation_color_img(np_out0,
                              np_in0,
                              nstride:int):
#  print(np_in0.shape, np_out0.shape)
  shape_np_out0 = np_out0.shape
  np_out0 = cv2.resize(np_out0, None, fx=nstride, fy=nstride)
  if shape_np_out0[2] == 1:
    assert np_out0.ndim == 2
    np_out0 = np_out0.reshape((np_out0.shape[0],np_out0.shape[1],1))
  ###
  np_res0 = np_in0.astype(numpy.float)
  mask_body = np_out0[:, :, 0]
#  print(mask_body.shape)
  np_res0[:, :, 0] = np_res0[:, :, 0] * mask_body
  np_res0[:, :, 1] = np_res0[:, :, 1] * mask_body
  np_res0[:, :, 2] = np_res0[:, :, 2] * mask_body
  if np_out0.shape[2] > 1:
    mask_bra = np_out0[:, :, 1]
    np_res0[:, :, 0] = np_res0[:, :, 0] * (1 - mask_bra) + mask_bra * 255.0
    np_res0[:, :, 1] = np_res0[:, :, 1] * (1 - mask_bra)
    np_res0[:, :, 2] = np_res0[:, :, 2] * (1 - mask_bra)
  np_res0 = numpy.clip(np_res0, 0.0, 255.0).astype(numpy.uint8)
  return np_res0
'''