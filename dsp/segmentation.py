import numpy, glob, cv2, json
import my_dnn_util.util_torch as my_torch
import my_dnn_util.util as my_util
import my_dnn_util.dsp.util as my_dsp

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