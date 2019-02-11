import cv2, numpy, json, random, glob
import torch
import my_dnn_util.dsp.util as my_dsp
import my_dnn_util.util as my_util
import my_dnn_util.util_torch as my_torch

def get_dict_info_from_heatmap(np_heatmap, face_rad, list_key_name, mag, thres=0.3):
  assert np_heatmap.shape[2] == len(list_key_name)
  list_pos_key = my_util.get_peaks(list_key_name, np_heatmap, mag, thres=thres)
  dict_info1 = {}
  dict_info1['person0'] = {}
  dict_info1['person0']['rad_head'] = face_rad
  for ipos, pos in enumerate(list_pos_key):
    if not list_pos_key[ipos][2] is 2:
      continue
    key_name = list_key_name[ipos]
    dict_info1['person0'][key_name] = list_pos_key[ipos]
  return dict_info1


def pose_detection_from_scratch(np_img,face_rad,list_key_name,net_cpm_scratch):
  mag = 16 / face_rad  # magnify such that face is approx. 12pix
  np_in_img = cv2.resize(np_img, (int(mag * np_img.shape[1]), int(mag * np_img.shape[0])))
  np_in_img = my_util.get_image_npix(np_in_img, net_cpm_scratch.npix, mag=1)
  np_in_img = np_in_img.reshape([1] + list(np_in_img.shape))
  #####
  vpt_in_img = my_torch.np2pt_img(np_in_img, 1.0 / 255.0, offset=0.0, requires_grad=False)
  with torch.no_grad():
    pt_out0 = net_cpm_scratch.forward(vpt_in_img)
  np_heatmap = numpy.moveaxis(pt_out0.data.numpy(), 1, 3)
  np_heatmap = np_heatmap.reshape(np_heatmap.shape[1:])
  dict_info1 = get_dict_info_from_heatmap(np_heatmap,
                                          face_rad,
                                          list_key_name, mag)
  return dict_info1


def pose_detection_from_head(np_img, dict_info0, list_key_name, net_cpm_head,thres=0.3):
  if not "person0" in dict_info0:
    return dict_info0
  if not "rad_head" in dict_info0["person0"]:
    return dict_info0
  net_cpm_head.eval()
  mag = 16 / dict_info0["person0"]["rad_head"]  # magnify such that face is approx. 12pix
  np_in_img = cv2.resize(np_img, (int(mag * np_img.shape[1]), int(mag * np_img.shape[0])))
  np_in_img = my_util.get_image_npix(np_in_img, net_cpm_head.npix, mag=1)
  np_in_img = np_in_img.reshape([1] + list(np_in_img.shape))
  #####
  np_in_wht = numpy.zeros((1, np_in_img.shape[1], np_in_img.shape[2], 1), dtype=numpy.float32)
  if "kp_head" in dict_info0["person0"]:
    my_util.gauss_keypoint(np_in_wht[0], 0,
                           dict_info0["person0"]["kp_head"][0] * mag,
                           dict_info0["person0"]["kp_head"][1] * mag,
                           16 * 0.5)
#  my_util.view_batch(np_in_img, np_in_wht, 1)
  #####
  vpt_in_img = my_torch.np2pt_img(np_in_img, 2.0 / 255.0, offset=-1.0, requires_grad=False)
  vpt_in_wht = my_torch.np2pt_img(np_in_wht, 1.0, offset=0.0, requires_grad=False)
  vpt_in = torch.cat((vpt_in_img, vpt_in_wht), dim=1)
  with torch.no_grad():
    pt_out0 = net_cpm_head.forward(vpt_in)
  np_heatmap = numpy.moveaxis(pt_out0.data.numpy(), 1, 3)
  np_heatmap = np_heatmap.reshape(np_heatmap.shape[1:])
  dict_info1 = get_dict_info_from_heatmap(np_heatmap,
                                          dict_info0["person0"]['rad_head'],
                                          list_key_name, mag,
                                          thres=thres)
  return dict_info1


##############################################################################################################
## batches for training CNN

class BatchesScratch:
  def __init__(self,list_path_dir_img, nbatch):
    list_json = []
    for path_dir_img in list_path_dir_img:
      list_json += glob.glob(path_dir_img + "/*.json")
    self.bd = my_util.BatchDispenser(nbatch, list_json)
    print("number of images:", len(self.bd.list_path_in))

  def get_batch_np(self, list_key, nstride):
    size_img = 256
    nblk = size_img // nstride
    nch_out = len(list_key)
    np_batch_in0 = numpy.empty((0,size_img,size_img,3),dtype=numpy.uint8)
    np_batch_tg0 = numpy.empty((0,nblk,nblk,nch_out),dtype=numpy.float32)
    ####
    for path_json in self.bd.get_batch_path():
      dict_info = json.load(open(path_json,"r"))
      np_img_org = cv2.imread(my_util.path_img_same_name(path_json))
      ####
      rot_mat,scale = my_dsp.get_affine(dict_info,(np_img_org.shape[1],np_img_org.shape[0]),(size_img,size_img))
      gauss_rad =  0.5 * dict_info["person0"]["face_rad"] * scale
      ####
      np_img_in = cv2.warpAffine(np_img_org, rot_mat, (size_img,size_img), flags=cv2.INTER_CUBIC)
      assert np_img_in.shape == (nblk * nstride, nblk * nstride, 3)
      ####
      np_wht_out = my_dsp.input_kp(list_key,dict_info,(size_img,size_img),
                                rot_mat,
                                gauss_rad)
      np_wht_out = np_wht_out[:,::nstride,::nstride,:]
      ####
      np_batch_tg0 = numpy.vstack((np_batch_tg0,np_wht_out.reshape(1,nblk,nblk,nch_out)))
      np_batch_in0 = numpy.vstack((np_batch_in0,np_img_in.reshape(1,size_img,size_img,3)))
    return np_batch_in0, np_batch_tg0

  def get_batch_vpt(self, list_key:list, nstride:int, requires_grad:bool):
    np_in, np_tg = self.get_batch_np(list_key, nstride)
    vpt_in = my_torch.np2pt_img(np_in,1.0/255.0,offset=0.0, requires_grad=requires_grad)
    vpt_tg = my_torch.np2pt_img(np_tg, 1.0, offset=0.0, requires_grad=False)
    return vpt_in, vpt_tg


class BatchesHead:
  def __init__(self,list_path_dir_img, nbatch):
    list_json = []
    for path_dir_img in list_path_dir_img:
      list_json += glob.glob(path_dir_img + "/*.json")
    self.bd = my_util.BatchDispenser(nbatch, list_json)
    print("number of images:", len(self.bd.list_path_in))

  def get_batch_np(self, list_key, nstride):
    size_img = 256
    nblk = size_img // nstride
    nch_out = len(list_key)
    np_batch_in_img = numpy.empty((0,size_img,size_img,3),dtype=numpy.uint8)
    np_batch_in_wht = numpy.empty((0, size_img, size_img, 1), dtype=numpy.float32)
    np_batch_tg_wht = numpy.empty((0,nblk,nblk,nch_out),dtype=numpy.float32)
    ####
    for path_json in self.bd.get_batch_path():
      dict_info = json.load(open(path_json,"r"))
      np_img_org = cv2.imread(my_util.path_img_same_name(path_json))
      ####
      rot_mat,scale = my_dsp.get_affine(dict_info,
                                        (np_img_org.shape[1], np_img_org.shape[0]),
                                        (size_img, size_img))
      gauss_rad =  0.5 * dict_info["person0"]["rad_head"] * scale
      ####
      np_in_img = cv2.warpAffine(np_img_org, rot_mat, (size_img,size_img), flags=cv2.INTER_CUBIC)
      assert np_in_img.shape == (nblk * nstride, nblk * nstride, 3)
      ####
      dict_info1 = dict_info
      if "kp_head" in dict_info1["person0"]:
        dict_info1["person0"]["kp_head"][0] += random.randint(-8,8)
        dict_info1["person0"]["kp_head"][1] += random.randint(-8,8)
      gauss_rad1 = gauss_rad*random.uniform(0.9,1.1)
      np_in_wht = my_dsp.input_kp(["kp_head"],dict_info1,(size_img,size_img),
                                rot_mat,
                                gauss_rad1)
      ####
      np_out_wht = my_dsp.input_kp(list_key,dict_info,(size_img,size_img),
                                rot_mat,
                                gauss_rad)
      np_out_wht = np_out_wht[:,::nstride,::nstride,:]
      ####
      np_batch_tg_wht = numpy.vstack((np_batch_tg_wht,np_out_wht.reshape(1,nblk,nblk,nch_out)))
      np_batch_in_wht = numpy.vstack((np_batch_in_wht,np_in_wht.reshape(1,size_img,size_img,1)))
      np_batch_in_img = numpy.vstack((np_batch_in_img,np_in_img.reshape(1,size_img,size_img,3)))
    return np_batch_in_img, np_batch_in_wht, np_batch_tg_wht

  def get_batch_vpt(self, list_key:list, nstride:int, requires_grad:bool):
    np_in_img, np_in_wht, np_tg_wht = self.get_batch_np(list_key, nstride)
    vpt_in_img = my_torch.np2pt_img(np_in_img,2.0/255.0,offset=-1.0, requires_grad=requires_grad)
    vpt_in_wht = my_torch.np2pt_img(np_in_wht,1.0,offset=0.0, requires_grad=requires_grad)
    vpt_in = torch.cat((vpt_in_img,vpt_in_wht),dim=1)
    vpt_tg = my_torch.np2pt_img(np_tg_wht, 1.0, offset=0.0, requires_grad=False)
    return vpt_in, vpt_tg