import torch.nn.functional
from my_dnn_util.detection import *

####################################

def affine_random_head(dict_info,size_input, size_output, target_rad, pos_offset=128, rot_offset=40):
  dict_prsn = dict_info["person0"]
  scale = target_rad / dict_prsn["rad_head"]
  scale *= random.uniform(0.5, 2.0)
  if 'kp_head' in dict_prsn:
    x0 = dict_prsn["kp_head"][0]
    y0 = dict_prsn["kp_head"][1]
    cnt = [x0+random.randint(-pos_offset,pos_offset),
           y0 + random.randint(-pos_offset, pos_offset)]  # x,y
  else:
    cnt = [size_input[0] * 0.5, + size_input[1] * 0.5]
  rot = random.randint(-rot_offset, rot_offset)
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
      np_bgr0 = cv2.imread(my_util.path_img_same_name(path_json))
      np_bgr0 = my_util.pca_color_augmentation(np_bgr0)
      rot_mat, scale = affine_random_head(dict_info,
                                          (np_bgr0.shape[1], np_bgr0.shape[0]), (size_img, size_img),
                                          target_rad=nstride*0.72, pos_offset=128, rot_offset=40)
      np_bgr1 = cv2.warpAffine(np_bgr0, rot_mat, (size_img, size_img), flags=cv2.INTER_CUBIC)
      np_tgc, np_tgl, np_msk = my_dsp.input_detect("kp_head",dict_info,nblk,nstride,rot_mat,scale)
      ####
      np_batch_in = numpy.vstack((np_batch_in, np_bgr1.reshape(1, size_img, size_img, 3)))
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

def detect_head(dict_info0, img_bgr, net_detect):
  if "person0" in dict_info0:
    if "rad_head" in dict_info0["person0"] and \
        "kp_head" in dict_info0["person0"] and \
        dict_info0["person0"]["kp_head"][2] == 2:
      return dict_info0

  list_res = detect_multires(img_bgr,net_detect)
  if len(list_res) == 0:
    return {}
  dict_info0["person0"] = {}
  dict_info0["person0"]["rad_head"] = list_res[2]
  dict_info0["person0"]["kp_head"] = [list_res[0], list_res[1], 2]
  return dict_info0