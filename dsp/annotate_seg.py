from OpenGL.GL import *
from OpenGL.GLUT import *
import cv2, glob, random, json, time, argparse, yaml

import my_dnn_util.util as my_util
import my_dnn_util.util_gl as my_gl
import my_dnn_util.dsp.util as my_dsp


########################
list_path_json = []
ind_list_path_json = 0
img_size_info = (250, 250, 1.0, 1.0, -1)
dict_info = {}
mode_prsn = "person0"
list_name_seg = None
ind_name_seg = 0
mode_edit = ""
iloop_selected = -1
ivtx_selected = -1
mouse_button = None
modelseg = None

def load_img_info(ioffind):
  global ind_list_path_json, img_size_info, dict_info, iloop_selected
  nimg = len(list_path_json)
  ind_list_path_json = (ind_list_path_json + ioffind + nimg) % nimg
  path_img0 = my_util.path_img_same_name(list_path_json[ind_list_path_json])
  print(ind_list_path_json,path_img0)
  #####
  id_tex_org = img_size_info[4]
  if id_tex_org is not None and glIsTexture(id_tex_org):
    glDeleteTextures(id_tex_org)
  np_img_bgr = cv2.imread(path_img0)
  img_size_info = my_gl.load_texture(np_img_bgr)
  glutReshapeWindow(img_size_info[0], img_size_info[1])
  ####
  path_json = path_img0.rsplit(".",1)[0] + ".json"
  dict_info = {}
  if os.path.isfile(path_json):
    with open(path_json,"r") as file0:
      dict_info = json.load(file0)
  if not "person0" in dict_info:
    dict_info["person0"] = {}
  if not "face_rad" in dict_info["person0"]:
    dict_info["person0"]["face_rad"] = 40.0
  iloop_selected = -1
  if list_name_seg[ind_name_seg] in dict_info["person0"]:
    iloop_selected = 0
  #####
#  key = cv2.waitKey(-1)

#####################################
def display():
  name_seg = list_name_seg[ind_name_seg]
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  my_gl.set_view_trans(img_size_info)
  my_gl.draw_img(img_size_info)
  my_dsp.draw_annotation_bbox(dict_info["person0"])
  my_dsp.draw_annotation_segmentation(dict_info["person0"],
                                     selected_loop=iloop_selected,
                                     name_seg=name_seg)
  my_dsp.draw_keypoint_circle(dict_info["person0"],1.0,(255,  0,  0),2,"keypoint_head")
#  util.draw_annotation_keypoint(dict_info["person0"])
  ####
  glDisable(GL_TEXTURE_2D)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  my_gl.glut_print(-0.9, -1.0, GLUT_BITMAP_HELVETICA_18, mode_prsn+"  "+name_seg+" "+mode_edit, (0, 0, 0))
  glutSwapBuffers()

def reshape(width, height):
  glViewport(0, 0,width, height)

def keyboard(bkey, x, y):
  global ind_list_path_json, img_size_info, mode_edit, iloop_selected, ivtx_selected
  name_seg = list_name_seg[ind_name_seg]
  key = bkey.decode("utf-8")
#  print("keydown",key)
  if key == 'q':
    exit()
  if key == ' ':
    mod = glutGetModifiers()
    if mod == 0:
      load_img_info(1)
    elif mod == 1:
      load_img_info(-1)
  if key == "0":
    mode_edit = 'bbox'
  if key == '1':
    mode_edit = 'loop'
    if not name_seg in dict_info[mode_prsn]:
      dict_info[mode_prsn][name_seg] = [[]]
    else:
      dict_info[mode_prsn][name_seg].append([])
    iloop_selected = 0
  if key == '2':
    mode_edit = 'move_delete_add'
    iloop_selected = -1
    ivtx_selected = -1
  if key == 'v':
    mag = 16 / dict_info["person0"]["face_rad"]
    path_img0 = my_util.path_img_same_name(list_path_json[ind_list_path_json])
    list_loop = modelseg.get_loop(path_img0,mag,name_seg)
    iloop_selected = -1 if len(list_loop)==0 else 0
    dict_info[mode_prsn][name_seg] = list_loop
  if key == 'd':
    if name_seg in dict_info[mode_prsn]:
      if 0 <= iloop_selected < len(dict_info[mode_prsn][name_seg]):
        dict_info[mode_prsn][name_seg].pop(iloop_selected)
        nloop = len(dict_info[mode_prsn][name_seg])
        if nloop == 0:
          iloop_selected = -1
        else:
          iloop_selected = (iloop_selected+nloop-1)%nloop
  '''
  if key == 't':
    if name_seg in dict_info["person0"]:
      if iloop_selected >= 0 and iloop_selected < len(dict_info["person0"]):
        loop_out = dfm2.simplify_polyloop(dict_info["person0"][name_seg][iloop_selected], 2.0)
        dict_info["person0"][name_seg][iloop_selected] = loop_out
  '''
  if key == 's':
    if name_seg in dict_info["person0"]:
      list_loop = dict_info["person0"][name_seg]
      list_loop_new = []
      for loop in list_loop:
        if len(loop) == 0:
          print("deleting new loop")
        else:
          list_loop_new.append(loop)
      dict_info["person0"][name_seg] = list_loop_new
    dict_info["saved_time"] = time.time()
    path_json = list_path_json[ind_list_path_json]
    with open(path_json,"w") as file0:
      json.dump(dict_info,file0,indent=2)

  glutPostRedisplay()

def mouse(button, state, x, y):
  global rect_drag, mode_edit, ivtx_selected, iloop_selected, mouse_button
  name_seg = list_name_seg[ind_name_seg]
  xy1 = my_gl.get_img_coord((x, y), img_size_info)
  mouse_button = button
  if mode_edit == 'bbox':
    if state == GLUT_DOWN:
      dict_info[mode_prsn]['bbox'] = [xy1[0], xy1[1], 5, 5]
  if mode_edit == 'loop':
    if state == GLUT_DOWN:
      if not name_seg in dict_info[mode_prsn]:
        dict_info[mode_prsn][name_seg] = [[]]
      iloop= len(dict_info[mode_prsn][name_seg]) - 1
      dict_info[mode_prsn][name_seg][iloop].extend([int(xy1[0]), int(xy1[1])])
  if mode_edit == 'move_delete_add':
    if state == GLUT_DOWN:
      if button==GLUT_LEFT_BUTTON:
        iloop_selected, ivtx_selected = my_util.pick_loop_vertex(xy1, name_seg, dict_info[mode_prsn])
        print(iloop_selected, ivtx_selected)
      if button==GLUT_MIDDLE_BUTTON:
        iloop_selected, ivtx_selected = my_util.pick_loop_vertex(xy1, name_seg, dict_info[mode_prsn])
        print(iloop_selected, ivtx_selected)
        if not ivtx_selected == -1:
          loop = dict_info[mode_prsn][name_seg][iloop_selected]
          loop.pop(ivtx_selected*2+1)
          loop.pop(ivtx_selected*2+0)
          ivtx_selected = -1
      elif button == GLUT_RIGHT_BUTTON:
        iloop_selected, ivtx_selected, xy3 = my_util.pick_loop_edge(xy1, name_seg, dict_info[mode_prsn])
        if not ivtx_selected == -1:
          loop = dict_info[mode_prsn][name_seg][iloop_selected]
          loop.insert(ivtx_selected*2+0,xy1[0])
          loop.insert(ivtx_selected*2+1,xy1[1])
    elif state == GLUT_UP:
      ivtx_selected = -1
  glutPostRedisplay()

def motion(x, y):
  xy = my_gl.get_img_coord((x,y),img_size_info)
  name_seg = list_name_seg[ind_name_seg]
  if mode_edit == 'bbox' and "bbox" in dict_info[mode_prsn]:
    dict_info[mode_prsn]["bbox"][2] = abs(xy[0] - dict_info[mode_prsn]["bbox"][0])
    dict_info[mode_prsn]["bbox"][3] = abs(xy[1] - dict_info[mode_prsn]["bbox"][1])
  if mode_edit == 'move_delete_add':
    if mouse_button == GLUT_LEFT_BUTTON:
      if name_seg in dict_info[mode_prsn]:
        if iloop_selected >= 0 and iloop_selected<len(dict_info[mode_prsn][name_seg]):
          loop = dict_info[mode_prsn][name_seg][iloop_selected]
          if ivtx_selected>=0 and ivtx_selected*2<len(loop):
            loop[ivtx_selected*2+0] = xy[0]
            loop[ivtx_selected*2+1] = xy[1]
  glutPostRedisplay()

def mySpecialFunc(key,x,y):
  global iloop_selected, ind_name_seg
  name_seg = list_name_seg[ind_name_seg]
  if key == GLUT_KEY_UP:
    if name_seg in dict_info[mode_prsn]:
      if len(dict_info[mode_prsn][name_seg]) > 0:
        iloop_selected = (iloop_selected+1)%len(dict_info[mode_prsn][name_seg])
      else:
        iloop_selected = -1
    else:
      iloop_selected = -1
  if key == GLUT_KEY_RIGHT:
    if len(list_name_seg) > 0:
      ind_name_seg = (ind_name_seg+1)%(len(list_name_seg))
    else:
      ind_name_seg = -1
  glutPostRedisplay()

def main():
  global list_path_json, ind_list_path_json, modelseg, list_name_seg
  # GLUT Window Initialization
  glutInit()
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)  # zBuffer
  glutInitWindowSize(600, 600)
  glutInitWindowPosition(100, 100)
  glutCreateWindow("Visualizzatore_2.0")
  # Register callbacks
  glutReshapeFunc(reshape)
  glutDisplayFunc(display)
  glutMouseFunc(mouse)
  glutMotionFunc(motion)
  glutKeyboardFunc(keyboard)
  glutSpecialFunc(mySpecialFunc)
  ####
  parser = argparse.ArgumentParser(description="train convolutional pose machine",
                                   add_help=True)
  parser.add_argument('--list_name_seg', required=True, nargs="+")
  parser.add_argument('--path_dir_img', help='input image dire', required=True)
  args = parser.parse_args()
  ####
  list_name_seg = args.list_name_seg
  print(list_name_seg)
  list_path_json = glob.glob(args.path_dir_img+"/*.json")
  list_path_json_annotated = my_dsp.list_annotated_and(list_path_json, list_name_seg)
  print(len(list_path_json_annotated),"/",len(list_path_json))
  random.shuffle(list_path_json)
  list_path_json = my_dsp.arrange_old_new_json(list_path_json)
  print(list_path_json)
  ind_list_path_json = 0
  load_img_info(0)
  ####
  glutMainLoop()

if __name__ == "__main__":
  main()
