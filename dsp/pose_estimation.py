import cv2, numpy
import torch
import my_dnn_util.util as my_util
import my_dnn_util.util_torch as my_torch



def get_dict_info_from_heatmap(np_heatmap, face_rad, list_key_name, mag, thres=0.3):
  assert np_heatmap.shape[2] == len(list_key_name)
  list_pos_key = my_util.get_peaks(list_key_name, np_heatmap, mag, thres=thres)
  dict_info1 = {}
  dict_info1['person0'] = {}
  dict_info1['person0']['face_rad'] = face_rad
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
  net_cpm_head.eval()
  mag = 16 / dict_info0["person0"]["face_rad"]  # magnify such that face is approx. 12pix
  np_in_img = cv2.resize(np_img, (int(mag * np_img.shape[1]), int(mag * np_img.shape[0])))
  np_in_img = my_util.get_image_npix(np_in_img, net_cpm_head.npix, mag=1)
  np_in_img = np_in_img.reshape([1] + list(np_in_img.shape))
  #####
  np_in_wht = numpy.zeros((1, np_in_img.shape[1], np_in_img.shape[2], 1), dtype=numpy.float32)
  if "keypoint_head" in dict_info0["person0"]:
    my_util.gauss_keypoint(np_in_wht[0], 0,
                           dict_info0["person0"]["keypoint_head"][0] * mag,
                           dict_info0["person0"]["keypoint_head"][1] * mag,
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
                                          dict_info0["person0"]['face_rad'],
                                          list_key_name, mag,
                                          thres=thres)
  return dict_info1