import cv2, numpy, json, time, random
from OpenGL.GL import *
from OpenGL.GLUT import *
import my_dnn_util.util_gl as my_gl

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


def cv2_draw_annotation(np_img0,dict_info,list_key,dict_key_prop):
  face_rad = dict_info["person0"]["face_rad"]
  for ikey, key in enumerate(list_key):
    if key in dict_info["person0"]:
      pos_key = dict_info["person0"][key]
      prop = dict_key_prop[key]
      color0 = prop[0:3]
      rad0 = prop[3]*face_rad
      width = int(prop[4])
      cv2.circle(np_img0, (int(pos_key[0]), int(pos_key[1])), int(rad0), color0, width)

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

def draw_annotation_keypoint(dict_info, dict_kp_draw_prop):
  for key in dict_info:
    if not key.startswith('keypoint_'):
      continue
    if key in dict_kp_draw_prop:
      prop = dict_kp_draw_prop[key]
      draw_keypoint_circle(dict_info,prop[3], (prop[2],prop[1],prop[0]),prop[4],key)
    else:
      draw_keypoint_circle(dict_info, 0.1, (0,0,0), 2.0, key)
  ####
  draw_keypoint_line(dict_info,(  0,  0,255), "keypoint_shoulderleft","keypoint_elbowleft")
  draw_keypoint_line(dict_info,(  0,255,  0), "keypoint_shoulderright","keypoint_elbowright")
  draw_keypoint_line(dict_info,(255,  0,255), "keypoint_elbowleft","keypoint_wristleft")
  draw_keypoint_line(dict_info,(255,255,  0), "keypoint_elbowright","keypoint_wristright")
  ####
  draw_keypoint_line(dict_info,(  0,  0,255), "keypoint_hipleft","keypoint_kneeleft")
  draw_keypoint_line(dict_info,(  0,255,  0), "keypoint_hipright","keypoint_kneeright")
  draw_keypoint_line(dict_info,(255,  0,255), "keypoint_kneeleft","keypoint_ankleleft")
  draw_keypoint_line(dict_info,(255,255,  0), "keypoint_kneeright","keypoint_ankleright")


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
