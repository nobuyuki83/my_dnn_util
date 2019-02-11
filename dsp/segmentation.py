import numpy, glob, cv2, json
import torch
import my_dnn_util.util_torch as my_torch
import my_dnn_util.util as my_util
import my_dnn_util.dsp.util as my_dsp

def segmentation_map_from_pose(img_org, dict_info1, list_kp, net_seg):
  if not "person0" in dict_info1:
    return img_org, numpy.zeros((img_org.shape[0],img_org.shape[1],1)), 1.0
  if not "rad_head" in dict_info1["person0"]:
    return img_org, numpy.zeros((img_org.shape[0],img_org.shape[1],1)), 1.0
  net_seg.eval()
  mag = 16 / dict_info1["person0"]["rad_head"]  # magnify such that face is approx. 12pix
  np_in_img = cv2.resize(img_org, (int(mag * img_org.shape[1]), int(mag * img_org.shape[0])))
  np_in_img = my_util.get_image_npix(np_in_img, net_seg.npix, mag=1)
  np_in_img = np_in_img.reshape([1] + list(np_in_img.shape))
  ####
  np_in_wht = numpy.zeros((1, np_in_img.shape[1], np_in_img.shape[2], len(list_kp)), dtype=numpy.float32)
  for ikp, kp in enumerate(list_kp):
    if not kp in dict_info1["person0"]:
      continue
    my_util.gauss_keypoint(np_in_wht[0], ikp,
                           dict_info1["person0"][kp][0] * mag,
                           dict_info1["person0"][kp][1] * mag,
                           16 * 0.5)
  vpt_in_img = my_torch.np2pt_img(np_in_img, scale=2.0 / 255.0, offset=-1.0, requires_grad=False)
  vpt_in_wht = my_torch.np2pt_img(np_in_wht, scale=1.0, offset=0.0, requires_grad=False)
  vpt_in = torch.cat((vpt_in_img, vpt_in_wht), dim=1)
  net_seg.eval()
  with torch.no_grad():
    vpt_out = net_seg.forward(vpt_in)
  np_out = numpy.moveaxis(vpt_out.data.numpy(), 1, 3)
  #    my_util.view_batch(np_in_img,np_out,nstride=1)
  np_out = np_out.reshape(np_out.shape[1:])
  np_in_img = np_in_img.reshape(np_in_img.shape[1:])
  return np_in_img, np_out, mag

##############################################################################################################
## batches for training CNN

class BatchesScratch:
  def __init__(self,path_dir_img, nbatch, list_name_seg):
    self.list_name_seg = list_name_seg
    list_path_json0 = glob.glob(path_dir_img + "/*.json")
    list_path_json0 = my_dsp.list_annotated_and(list_path_json0, list_name_seg)
    self.bd = my_util.BatchDispenser(nbatch,list_path_json0)
    print("number of images:", len(self.bd.list_path_in))

  def get_batch_np(self):
    nch_out = len(self.list_name_seg)
    size_img = 256
    np_batch_in0 = numpy.empty((0,size_img,size_img,3),dtype=numpy.uint8)
    np_batch_tg0 = numpy.empty((0,size_img,size_img,nch_out),dtype=numpy.float32)
    ####
    for path_json in self.bd.get_batch_path():
      dict_info = json.load(open(path_json,"r") )
      np_img0 = cv2.imread(my_util.path_img_same_name(path_json))
      rot_mat,scale = my_dsp.get_affine(dict_info,(np_img0.shape[1],np_img0.shape[0]),(size_img,size_img))
      ####
      np_img1 = cv2.warpAffine(np_img0, rot_mat, (size_img,size_img), flags=cv2.INTER_CUBIC)
      np_anno0 = my_dsp.input_seg(self.list_name_seg, dict_info,
                                  (size_img,size_img),rot_mat,np_img0.shape[0:2])
      #####
      np_batch_in0 = numpy.vstack((np_batch_in0,np_img1.reshape(1,size_img,size_img,3)))
      np_batch_tg0 = numpy.vstack((np_batch_tg0,np_anno0))
    np_batch_in0 = np_batch_in0.astype(numpy.uint8)
    return np_batch_in0, np_batch_tg0

  def get_batch_vpt(self,requires_grad=False):
    np_batch_in0, np_batch_tg0 = self.get_batch_np()
    vpt_batch_in0 = my_torch.np2pt_img(np_batch_in0,scale=1.0/255.0,offset=0.0,requires_grad=requires_grad)
    vpt_batch_tg0 = my_torch.np2pt_img(np_batch_tg0,scale=1.0,offset=0.0,requires_grad=False)
    return vpt_batch_in0, vpt_batch_tg0


class BatchesPose:
  def __init__(self,list_path_dir_img, nbatch, list_name_seg, list_name_kp):
    self.list_name_seg = list_name_seg
    self.list_name_kp = list_name_kp
    list_path_json0 = []
    for path_dir_img in list_path_dir_img:
      list_path_json0 += glob.glob(path_dir_img + "/*.json")
    list_path_json0 = my_dsp.list_annotated_and(list_path_json0, list_name_seg)
    self.bd = my_util.BatchDispenser(nbatch,list_path_json0)
    print("number of images:", len(self.bd.list_path_in))

  def get_batch_np(self):
    nch_out = len(self.list_name_seg)
    nch_in_wht = len(self.list_name_kp)
    size_img = 256
    np_batch_in_img = numpy.empty((0,size_img,size_img,3),dtype=numpy.uint8)
    np_batch_in_wht = numpy.empty((0,size_img,size_img,nch_in_wht),dtype=numpy.float32)
    np_batch_tg_seg = numpy.empty((0,size_img,size_img,nch_out),dtype=numpy.float32)
    ####
    for path_json in self.bd.get_batch_path():
      dict_info = json.load(open(path_json,"r") )
      np_img0 = cv2.imread(my_util.path_img_same_name(path_json))
      rot_mat,scale = my_dsp.get_affine(dict_info,(np_img0.shape[1],np_img0.shape[0]),(size_img,size_img))
      ####
      np_in_img = cv2.warpAffine(np_img0, rot_mat, (size_img,size_img), flags=cv2.INTER_CUBIC)
      gauss_rad = 0.5*dict_info["person0"]["rad_head"]*scale
      np_in_wht = my_dsp.input_kp(self.list_name_kp,dict_info,
                                  (size_img, size_img), rot_mat, gauss_rad)
      np_tg_seg = my_dsp.input_seg(self.list_name_seg, dict_info,
                                  (size_img,size_img),rot_mat,np_img0.shape[0:2])
      #####
      np_batch_in_img = numpy.vstack((np_batch_in_img,np_in_img.reshape(1,size_img,size_img,3)))
      np_batch_in_wht = numpy.vstack((np_batch_in_wht,np_in_wht))
      np_batch_tg_seg = numpy.vstack((np_batch_tg_seg,np_tg_seg))
    np_batch_in_img = np_batch_in_img.astype(numpy.uint8)
    return np_batch_in_img, np_batch_in_wht, np_batch_tg_seg

  def get_batch_vpt(self,requires_grad=False):
    np_in_img, np_in_wht, np_tg_seg = self.get_batch_np()
    vpt_in_img = my_torch.np2pt_img(np_in_img,scale=2.0/255.0,offset=-1.0,requires_grad=requires_grad)
    vpt_in_wht = my_torch.np2pt_img(np_in_wht,scale=1.0,offset=0.0,requires_grad=requires_grad)
    vpt_in = torch.cat((vpt_in_img,vpt_in_wht),dim=1)
    vpt_tg = my_torch.np2pt_img(np_tg_seg,scale=1.0,offset=0.0,requires_grad=False)
    return vpt_in, vpt_tg