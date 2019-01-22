
import cv2, math, hashlib, os, numpy, argparse, json, time
from scipy.ndimage.filters import maximum_filter


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

def path_img_same_name(path_json:str):
  if os.path.isfile(path_json.rsplit(".", 1)[0] + ".jpg"):
    path_img = path_json.rsplit(".", 1)[0] + ".jpg"
  else:
    path_img = path_json.rsplit(".", 1)[0] + ".png"
  return path_img

def pick_loop(xy1:list,key:str,dict_prsn:dict):
  iloop_selected = -1
  ivtx_selected = -1
  if not key in dict_prsn:
    return
  min_len = -1.0
  for iloop, loop in enumerate(dict_prsn[key]):
    for ivtx in range(len(loop) // 2):
      x0 = loop[ivtx * 2 + 0]
      y0 = loop[ivtx * 2 + 1]
      len0 = math.sqrt((x0 - xy1[0]) * (x0 - xy1[0]) + (y0 - xy1[1]) * (y0 - xy1[1]))
      if min_len < 0 or len0 < min_len:
        min_len = len0
        iloop_selected = iloop
        ivtx_selected = ivtx
  return iloop_selected,ivtx_selected

def list_annotated(list_path_json0,list_name_seg):
  list_path_json = []
  for path_json in list_path_json0:
    dict_info = json.load(open(path_json, "r"))
    if "person0" in dict_info:
      is_ok = True
      for name_seg in list_name_seg:
        if not name_seg in dict_info["person0"]:
          is_ok = False
      if is_ok:
        list_path_json.append(path_json)
  return list_path_json

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


def arrange_old_new_json(list_path_json):
  dict_bn_time = {}
  for path_json in list_path_json:
    if not os.path.isfile(path_json):
      dict_bn_time[path_json] = -time.time()
      continue
    with open(path_json,"r") as file0:
      dict_info0 = json.load(file0)
      if "saved_time" in dict_info0:
        dict_bn_time[path_json] = -dict_info0["saved_time"]
      else:
        dict_bn_time[path_json] = -time.time()
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



def get_peaks(list_key,np_out0,mag):
  pos_key = [[0, 0, 0]] * len(list_key)
  for ikey, key in enumerate(list_key):
    local_max = maximum_filter(np_out0[:, :, ikey], footprint=numpy.ones((5, 5))) == np_out0[:, :, ikey]
    local_max = local_max * (np_out0[:, :, ikey] > 0.2)
    peaks = ((numpy.array(numpy.nonzero(local_max)[::-1]).T) * (2.0 / mag)).astype(numpy.int)
    if peaks.shape[0] == 1:
      pos_key[ikey] = [peaks[0][0], peaks[0][1], 2]
  return pos_key


def get_segmentation_color_img(np_out0,
                              np_in0,
                              nstride:int):
  np_out0 = cv2.resize(np_out0, None, fx=nstride, fy=nstride)
  ###
  np_res0 = np_in0.astype(numpy.float)
  mask_body = np_out0[:, :, 0]
  print(mask_body.shape)
  np_res0[:, :, 0] = np_res0[:, :, 0] * mask_body
  np_res0[:, :, 1] = np_res0[:, :, 1] * mask_body
  np_res0[:, :, 2] = np_res0[:, :, 2] * mask_body
  ####
  mask_bra = np_out0[:, :, 1]
  np_res0[:, :, 0] = np_res0[:, :, 0] * (1 - mask_bra) + mask_bra * 255.0
  np_res0[:, :, 1] = np_res0[:, :, 1] * (1 - mask_bra)
  np_res0[:, :, 2] = np_res0[:, :, 2] * (1 - mask_bra)
  np_res0 = numpy.clip(np_res0, 0.0, 255.0).astype(numpy.uint8)
  return np_res0



