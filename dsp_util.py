import cv2, numpy, json, time, random
from OpenGL.GL import *
from OpenGL.GLUT import *
import my_dnn_util.util_gl as my_gl
import my_dnn_util.util as my_util

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# person dataset from here

def list_annotated_and(list_path_json0, list_name_anno):
  list_path_json = []
  for path_json in list_path_json0:
    dict_info = json.load(open(path_json, "r"))
    if "person0" in dict_info:
      is_ok = True
      for name_seg in list_name_anno:
        if not name_seg in dict_info["person0"]:
          is_ok = False
      if is_ok:
        list_path_json.append(path_json)
  return list_path_json


def list_annotated_or(list_path_json0, list_name_anno):
  list_path_json = []
  for path_json in list_path_json0:
    dict_info = json.load(open(path_json, "r"))
    if "person0" in dict_info:
      is_ok = False
      for name_seg in list_name_anno:
        if name_seg in dict_info["person0"]:
          is_ok = True
      if is_ok:
        list_path_json.append(path_json)
  return list_path_json

def get_affine(dict_info, size_input, size_output, rot_range=(-40,40), mag_range=(0.8, 1,2)):
  dict_prsn = dict_info["person0"]
  scale = 16 / dict_prsn["face_rad"]
  scale *= random.uniform(mag_range[0], mag_range[1])
  cnt = [size_input[0] * 0.5, + size_input[1] * 0.5]
  if "bbox" in dict_prsn:
    bbox = dict_prsn["bbox"]
    assert type(bbox) == list and len(bbox) == 4
    cnt[0] = bbox[0] + bbox[2] * 0.5
    cnt[1] = bbox[1] + bbox[3] * 0.5
  rot = random.randint(rot_range[0], rot_range[1])
  rot_mat = cv2.getRotationMatrix2D(tuple([cnt[0], cnt[1]]), rot, scale)
  rot_mat[0][2] += size_output[0] * 0.5 - cnt[0] + random.randint(int(-8 / scale), int(+8 / scale))
  rot_mat[1][2] += size_output[1] * 0.5 - cnt[1] + random.randint(int(-8 / scale), int(+8 / scale))
  return rot_mat, scale

def input_kp(list_name_kp,dict_info,size_img_out,rot_mat,gauss_rad):
  nch_out = len(list_name_kp)
  np_wht0 = numpy.zeros((1,size_img_out[0],size_img_out[1], nch_out), dtype=numpy.float32)
  dict_prsn = dict_info["person0"]
  for ikey, key in enumerate(list_name_kp):
    if not key in dict_prsn: continue
    if not dict_prsn[key][2] == 2: continue
    cx0, cy0, iflg = dict_prsn[key]
    dx0 = rot_mat[0][0] * cx0 + rot_mat[0][1] * cy0 + rot_mat[0][2]
    dy0 = rot_mat[1][0] * cx0 + rot_mat[1][1] * cy0 + rot_mat[1][2]
    my_util.gauss_keypoint(np_wht0[0], ikey,
                           dx0, dy0, gauss_rad)
  return np_wht0

def input_seg(list_name_seg, dict_info, size_img_in, size_img_out, rot_mat):
  nch_out = len(list_name_seg)
  np_anno0 = numpy.zeros((1, size_img_out[0], size_img_out[1], nch_out), dtype=numpy.uint8)
  for iseg, name_seg in enumerate(list_name_seg):
    list_loop = dict_info["person0"][name_seg]
    list_np_seg = my_util.cv2_get_numpy_loop_array(list_loop)
    np_mask0 = numpy.zeros(size_img_in, dtype=numpy.uint8)
    cv2.fillPoly(np_mask0, list_np_seg, color=1.0)
    np_mask0 = cv2.warpAffine(np_mask0, rot_mat, size_img_out, flags=cv2.INTER_CUBIC)
    np_anno0[0][:, :, iseg] = np_mask0
  return np_anno0


def input_kp_assembled(list_name_kp, dict_info, shape_img_out, rot_mat, shape_img_in, mask_size):
  np_anno0 = numpy.zeros((shape_img_out[0], shape_img_out[1]), dtype=numpy.uint8)
  for ikp, name_kp in enumerate(list_name_kp):
    if not name_kp in dict_info["person0"]:
      continue
    x, y, _ = dict_info["person0"][name_kp]
    list_loop = [my_util.get_random_polygon(x, y, mask_size)]
    list_np_seg = my_util.cv2_get_numpy_loop_array(list_loop)
    np_mask0 = numpy.zeros((shape_img_in[0], shape_img_in[1]), dtype=numpy.uint8)
    cv2.fillPoly(np_mask0, list_np_seg, color=1.0)
    np_mask0 = cv2.warpAffine(np_mask0, rot_mat, shape_img_out, flags=cv2.INTER_CUBIC)
    np_anno0 = np_anno0 + np_mask0 - np_anno0 * np_mask0  # computing or
  np_anno0 = np_anno0.reshape(1, shape_img_out[0], shape_img_out[1], 1)
  return np_anno0


def input_seg_assembled(list_name_seg, dict_info, shape_img_out, rot_mat, shape_img_in):
  np_anno0 = numpy.zeros((shape_img_out[0], shape_img_out[1]), dtype=numpy.uint8)
  for iseg, name_seg in enumerate(list_name_seg):
    if not name_seg in dict_info["person0"]:
      continue
    list_loop = dict_info["person0"][name_seg]
    list_np_seg = my_util.cv2_get_numpy_loop_array(list_loop)
    np_mask0 = numpy.zeros((shape_img_in[0], shape_img_in[1]), dtype=numpy.uint8)
    cv2.fillPoly(np_mask0, list_np_seg, color=1.0)
    np_mask0 = cv2.warpAffine(np_mask0, rot_mat, shape_img_out, flags=cv2.INTER_CUBIC)
    np_anno0[:, :] = np_anno0 + np_mask0 - np_anno0 * np_mask0  # computing or
  np_anno0 = np_anno0.reshape(1, shape_img_out[0], shape_img_out[1], 1)
  return np_anno0


#############################################################################################################################


def cv2_draw_annotation(np_img0,dict_info,list_key,dict_key_prop,list_edge_prop):
  face_rad = dict_info["person0"]["face_rad"]
  for ikey, key in enumerate(list_key):
    if key in dict_info["person0"]:
      pos_key = dict_info["person0"][key]
      prop = dict_key_prop[key]
      color0 = prop[0:3]
      rad0 = prop[3]*face_rad
      width = int(prop[4])
      cv2.circle(np_img0,
                 (int(pos_key[0]), int(pos_key[1])), int(rad0), color0, width)

  for edge in list_edge_prop:
    key0 = edge[0]
    key1 = edge[1]
    if key0 in dict_info["person0"] and key1 in dict_info['person0']:
      pos_key0 = dict_info["person0"][key0]
      pos_key1 = dict_info["person0"][key1]
      color0 = edge[2:5]
      cv2.line(np_img0,
               (int(pos_key0[0]), int(pos_key0[1])),
               (int(pos_key1[0]), int(pos_key1[1])),
               color0)

  if "bbox" in dict_info["person0"]:
    bbox = dict_info["person0"]["bbox"]
    cv2.rectangle(np_img0,
                  (int(bbox[0]),int(bbox[1])),
                  (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                  (255, 0, 0), 1)

  if "segmentation" in dict_info["person0"]:
    list_seg = dict_info["person0"]["segmentation"]
    for seg in list_seg:
      np_sgm = numpy.array(seg,dtype=numpy.int)
      np_sgm = np_sgm.reshape((-1,2))
      cv2.polylines(np_img0, [np_sgm], True, (0, 255, 255))

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


#####################################################################################
## OpenGL dependency from here

def draw_keypoint_circle(dict_info,head_ratio,color,width,key):
  if not "face_rad" in dict_info:
    return
  r0 = dict_info["face_rad"]
  if key in dict_info:
    if not dict_info[key][2] == 0:
       my_gl.drawCircle(dict_info[key],r0*head_ratio,color=color,width=width)

def draw_keypoint_line(dict_info,color,key0,key1):
  if key0 in dict_info and key1 in dict_info:
    if not dict_info[key0][2] == 0 and not dict_info[key1][2] ==0:
      my_gl.drawLine(dict_info[key0],dict_info[key1],color,width=2)

def draw_annotation_keypoint(dict_info, dict_kp_draw_prop,kp_edge_draw_prop):
  for key in dict_info:
    if not key.startswith('keypoint_'):
      continue
    if key in dict_kp_draw_prop:
      prop = dict_kp_draw_prop[key]
      draw_keypoint_circle(dict_info,prop[3], (prop[2],prop[1],prop[0]),prop[4],key)
    else:
      draw_keypoint_circle(dict_info, 0.1, (0,0,0), 2.0, key)
  ####
  for kp_edge in kp_edge_draw_prop:
    draw_keypoint_line(dict_info, (kp_edge[2], kp_edge[3], kp_edge[4]), kp_edge[0], kp_edge[1])

def draw_annotation_bbox(dict_info):
  if 'bbox' in dict_info:
    my_gl.drawRect(dict_info["bbox"],color=(255,0,0),width=1)

def draw_annotation_segmentation(dict_info,selected_loop:int,name_seg:str):
  if name_seg in dict_info:
    for iloop,loop in enumerate(dict_info[name_seg]):
      my_gl.drawPolyline(loop,color=(1,1,1),width=1)
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

