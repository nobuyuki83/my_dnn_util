import glob, numpy, cv2, json
import torch
import my_dnn_util.util as my_util
import my_dnn_util.dsp.util as my_dsp
import my_dnn_util.util_torch as my_torch


class Batches_MaskSeg_InKp:
  def __init__(self,path_dir_img:str, nbatch:int, list_seg_mask:list, list_kp_pose):
    list_path_json = glob.glob(path_dir_img + "/*.json")
    list_path_json = my_dsp.list_annotated_or(list_path_json,list_seg_mask)
    self.bd = my_util.BatchDispenser(nbatch, list_path_json)
    self.list_seg_mask = list_seg_mask
    self.list_kp_pose = list_kp_pose
    print("number of images:", len(self.bd.list_path_in))


  def get_batch_np(self, nstride:int):
    size_img = 224
    nblk = size_img // nstride
    np_batch_bgr0 = numpy.empty((0,size_img,size_img,3),dtype=numpy.uint8)
    np_batch_mask0 = numpy.empty((0, nblk, nblk, 1), dtype=numpy.float32)
    np_batch_kp0 = numpy.empty((0, nblk, nblk, len(self.list_kp_pose)), dtype=numpy.float32)
    ####
    for path_json in self.bd.get_batch_path():
      dict_info = json.load(open(path_json,"r"))
      np_img0 = cv2.imread(my_util.path_img_same_name(path_json))
      np_img0 = my_util.pca_color_augmentation(np_img0)
      rot_mat,scale = my_dsp.get_affine(dict_info,
                                        (np_img0.shape[1], np_img0.shape[0]),
                                        (size_img, size_img),
                                        npix_face_rad=24)
      ####
      np_img1 = cv2.warpAffine(np_img0, rot_mat, (size_img,size_img), flags=cv2.INTER_CUBIC)
      assert np_img1.shape == (nblk * nstride, nblk * nstride, 3)
      ####
      np_mask = my_dsp.input_seg(self.list_seg_mask,dict_info,
                                (size_img, size_img), rot_mat, np_img0.shape)
      np_kp = my_dsp.input_kp(self.list_kp_pose, dict_info, (size_img, size_img), rot_mat,
                              dict_info["person0"]["rad_head"]*scale*0.5)
      ####
      np_batch_bgr0 = numpy.vstack((np_batch_bgr0,np_img1.reshape(1,size_img,size_img,3)))
      np_batch_mask0 = numpy.vstack((np_batch_mask0, np_mask))
      np_batch_kp0 = numpy.vstack((np_batch_kp0, np_kp))
    np_batch_bgr0 = np_batch_bgr0.astype(numpy.uint8)
    return np_batch_bgr0, np_batch_mask0, np_batch_kp0

  def get_batch_vpt_real(self,nstride):
    np_batch_bgr, np_batch_mask, np_batch_kp = self.get_batch_np(nstride)
    vpt_batch_bgr = my_torch.np2pt_img(np_batch_bgr,2.0/255.0,offset=-1.0,requires_grad=False)
    vpt_batch_mask = my_torch.np2pt_img(np_batch_mask,1.0, offset=0.0,requires_grad=False)
    vpt_batch_kp = my_torch.np2pt_img(np_batch_kp,1.0,offset=0.0,requires_grad=False)
    return vpt_batch_bgr, vpt_batch_mask, vpt_batch_kp

  def get_batch_vpt_realfake(self,net_gen, requires_grad_fake=False):
    vpt_bgr, vpt_mask, vpt_kp  = self.get_batch_vpt_real(net_gen.nstride)
    vpt_bgr1 = vpt_bgr * (1 - vpt_mask)
    vpt_in = torch.cat((vpt_bgr1, vpt_mask, vpt_kp), 1)
    if requires_grad_fake:
      vpt_in.requires_grad = True
      vpt_out0 = net_gen.forward(vpt_in)
    else:
      with torch.no_grad():
        vpt_out0 = net_gen.forward(vpt_in)
      vpt_out0 = vpt_out0.detach()
    vpt_bgr_fake = vpt_out0 * vpt_mask + vpt_bgr * (1 - vpt_mask)
    ####
    vpt_out_real = torch.cat((vpt_bgr,vpt_kp),dim=1)
    vpt_out_fake = torch.cat((vpt_bgr_fake,vpt_kp),dim=1)
    vpt_tg_fake = my_torch.get_mask_ratio_vpt(vpt_mask)
    if torch.cuda.is_available():
      vpt_tg_fake = vpt_tg_fake.cuda()
    return vpt_out_real,vpt_out_fake,vpt_tg_fake



#####################################################################################
class Batches_MaskKp_InSegKp:
  def __init__(self, path_dir_img:str, nbatch:int,
               list_kp_mask: list,
               list_seg: list, list_kp_pose: list):
    list_path_json = glob.glob(path_dir_img + "/*.json")
    list_path_json = my_dsp.list_annotated_or(list_path_json, list_kp_mask)
    list_path_json = my_dsp.list_annotated_or(list_path_json, list_seg)
    self.bd = my_util.BatchDispenser(nbatch, list_path_json)
    self.list_kp_mask = list_kp_mask
    self.list_seg = list_seg
    self.list_kp_pose = list_kp_pose
    print("number of images:", len(self.bd.list_path_in))


  def get_batch_np(self, nstride:int):
    size_img = 224
    nblk = size_img // nstride
    np_batch_bgr0 = numpy.empty((0,size_img,size_img,3),dtype=numpy.uint8)
    np_batch_mask0 = numpy.empty((0, nblk, nblk, 1), dtype=numpy.float32)
    np_batch_seg0 = numpy.empty((0, nblk, nblk, 1), dtype=numpy.float32)
    np_batch_kp0 = numpy.empty((0, nblk, nblk, len(self.list_kp_pose)), dtype=numpy.float32)
    ####
    for path_json in self.bd.get_batch_path():
      dict_info = json.load(open(path_json,"r"))
      np_img0 = cv2.imread(my_util.path_img_same_name(path_json))
      np_img0 = my_util.pca_color_augmentation(np_img0)
      rot_mat,scale = my_dsp.get_affine(dict_info,
                                        (np_img0.shape[1], np_img0.shape[0]),
                                        (size_img, size_img),
                                        npix_face_rad=24)
      ####
      np_img1 = cv2.warpAffine(np_img0, rot_mat, (size_img,size_img), flags=cv2.INTER_CUBIC)
      assert np_img1.shape == (nblk * nstride, nblk * nstride, 3)
      ####
      np_mask = my_dsp.input_kp_assembled(self.list_kp_mask, dict_info,
                                          (size_img, size_img), rot_mat, np_img0.shape,
                                          16 / scale)
      np_seg = my_dsp.input_seg(self.list_seg,dict_info,
                                (size_img, size_img), rot_mat, np_img0.shape)
      np_kp = my_dsp.input_kp(self.list_kp_pose, dict_info, (size_img, size_img), rot_mat,
                              dict_info["person0"]["rad_head"] * scale)
      ####
      np_batch_bgr0 = numpy.vstack((np_batch_bgr0,np_img1.reshape(1,size_img,size_img,3)))
      np_batch_mask0 = numpy.vstack((np_batch_mask0, np_mask))
      np_batch_seg0 = numpy.vstack((np_batch_seg0, np_seg))
      np_batch_kp0 = numpy.vstack((np_batch_kp0, np_kp))
    np_batch_bgr0 = np_batch_bgr0.astype(numpy.uint8)
    '''
    for iseg in range(np_batch_kp0.shape[3]):
      np_batch_kp0[:,:,:,iseg] = np_batch_kp0[:,:,:,iseg]*np_batch_seg0[:,:,:,0]
    '''
    np_batch_kp0 = np_batch_kp0*np_batch_seg0
    np_s = numpy.sum(np_batch_kp0,axis=3)+1.0e-10
    np_batch_kp0 /= np_s.reshape(list(np_s.shape)+[1])
    return np_batch_bgr0, np_batch_mask0, np_batch_seg0, np_batch_kp0

  def get_batch_vpt_real(self,nstride):
    np_batch_bgr, np_batch_mask, np_batch_seg, np_batch_kp = self.get_batch_np(nstride)
    vpt_batch_bgr = my_torch.np2pt_img(np_batch_bgr,2.0/255.0,offset=-1.0,requires_grad=False)
    vpt_batch_mask = my_torch.np2pt_img(np_batch_mask,1.0, offset=0.0,requires_grad=False)
    vpt_batch_seg = my_torch.np2pt_img(np_batch_seg,1.0,offset=0.0,requires_grad=False)
    vpt_batch_kp = my_torch.np2pt_img(np_batch_kp,1.0,offset=0.0,requires_grad=False)
    return vpt_batch_bgr, vpt_batch_mask, vpt_batch_seg, vpt_batch_kp

  def get_batch_vpt_realfake(self,net_gen, requires_grad_fake=False):
    vpt_bgr_real, vpt_mask, vpt_seg, vpt_kp  = self.get_batch_vpt_real(net_gen.nstride)
    vpt_bgr_masked = vpt_bgr_real * (1 - vpt_mask)
    vpt_in = torch.cat((vpt_bgr_masked, vpt_mask, vpt_seg), 1)
    if requires_grad_fake:
      vpt_in.requires_grad = True
      vpt_out0 = net_gen.forward(vpt_in)
    else:
      with torch.no_grad():
        vpt_out0 = net_gen.forward(vpt_in)
      vpt_out0 = vpt_out0.detach()
    vpt_bgr_fake = vpt_out0 * vpt_mask + vpt_bgr_masked
    ####
#    vpt_out_real = torch.cat((vpt_bgr_real,vpt_seg,vpt_kp),dim=1)
#    vpt_out_fake = torch.cat((vpt_bgr_fake,vpt_seg,vpt_kp),dim=1)
    vpt_out_real = torch.cat((vpt_bgr_real,vpt_kp),dim=1)
    vpt_out_fake = torch.cat((vpt_bgr_fake,vpt_kp),dim=1)
    vpt_tg_fake = my_torch.get_mask_ratio_vpt(vpt_mask)
    if torch.cuda.is_available():
      vpt_tg_fake = vpt_tg_fake.cuda()
    return vpt_out_real,vpt_out_fake,vpt_tg_fake