import sys
import my_dnn_util.dsp.segmentation as my_seg
import my_dnn_util as my_util
import my_dnn_util.util_gl as my_gl

sys.path.append('delfem2/module_py')
import dfm2

def segmentation_loop_from_map(dict_info0, np_out, mag, list_name_seg):
  dict_info1 = dict_info0
  if not "person0" in dict_info1:
    return dict_info1
  #####
  assert len(list_name_seg) == np_out.shape[2]
  for iseg,name_seg in enumerate(list_name_seg):
    list_loop = dfm2.get_level_set(np_out[:, :, iseg].copy(), 1.0 / mag)
    for iloop in range(len(list_loop)):
      loop_out = dfm2.simplify_polyloop(list_loop[iloop], 3.0)
      list_loop[iloop] = loop_out
    for iloop in range(len(list_loop))[::-1]:
      area = my_gl.area_loop(list_loop[iloop]) * mag * mag
      if abs(area) < 90:
        del list_loop[iloop]
    if not name_seg in dict_info1["person0"]:
      dict_info1["person0"][name_seg] = list_loop
  return dict_info1

def segmentation_from_pose(dict_info0,img_org,list_kp_seg,net_seg,list_name_seg):
  img_seg_in, img_seg_out, mag = my_seg.segmentation_map_from_pose(img_org, dict_info0, list_kp_seg, net_seg)
  return segmentation_loop_from_map(dict_info0, img_seg_out, mag, list_name_seg)