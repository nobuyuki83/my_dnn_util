from OpenGL.GL import *
from OpenGL.GLUT import *
import cv2, glob, random, json, math, time, argparse, yaml
import my_dnn_util.util as my_util
import my_dnn_util.util_gl as my_gl
import my_dnn_util.dsp_util as my_dsp

########################
list_path_json = []
ind_list_path_json = 0
img_size_info = (250, 250, 1.0, 1.0, -1)
dict_info = {}
dict_config = None
mode_prsn = "person0"
mode_edit = ""

def load_img_info(ioffind):
  global ind_list_path_json, img_size_info, dict_info, mode_prsn
  nimg = len(list_path_json)
  ind_list_path_json = (ind_list_path_json + ioffind + nimg) % nimg
  path_json0 = list_path_json[ind_list_path_json]
  print(path_json0)
  path_img0 = my_util.path_img_same_name(path_json0)
  name_md5 = my_util.md5_hash(path_img0)
  assert name_md5 == path_img0.rsplit("/",1)[1].rsplit(".",1)[0]
  #####
  np_img_bgr = cv2.imread(path_img0)
  #####
  id_tex_org = img_size_info[4]
  if id_tex_org is not None and glIsTexture(id_tex_org):
    glDeleteTextures(id_tex_org)
  img_size_info = my_gl.load_texture(np_img_bgr)
  glutReshapeWindow(img_size_info[0], img_size_info[1])
  ####
  print(path_json0,ind_list_path_json,len(list_path_json))
  dict_info = {}
  if os.path.isfile(path_json0):
    with open(path_json0,"r") as file0:
      dict_info = json.load(file0)
  mode_prsn = "person0"
  if not "person0" in dict_info:
    dict_info["person0"] = {}
  if not "face_rad" in dict_info["person0"]:
    dict_info["person0"]["face_rad"] = 100.0


#####################################
def display():
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  my_gl.set_view_trans(img_size_info)
  my_gl.draw_img(img_size_info)
  my_dsp.draw_annotation_keypoint(dict_info[mode_prsn],dict_config['kp_draw_prop'])
  my_dsp.draw_annotation_bbox(dict_info[mode_prsn])
  ####
  glDisable(GL_TEXTURE_2D)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  my_gl.glut_print(-0.9, -1.0, GLUT_BITMAP_HELVETICA_18, mode_prsn+"  "+mode_edit, (0, 0, 0))
  glutSwapBuffers()

def reshape(width, height):
  glViewport(0, 0,width, height)

def keyboard(bkey, x, y):
  global ind_list_path_img, img_size_info, mode_edit
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
  if key == '-':
    dict_info[mode_prsn]["face_rad"] -= 3
  if key == '=':
    dict_info[mode_prsn]["face_rad"] += 3
  if key == "0":
    mode_edit = 'bbox'
  if key in dict_config['key2kp']:
    mode_edit = dict_config['key2kp'][key]
  if key == 'x':
    dict_info[mode_prsn].pop(mode_edit)
  if key == 's':
    dict_info["saved_time"] = time.time()
    path_json0 = list_path_json[ind_list_path_json]
    with open(path_json0,"w") as file0:
      json.dump(dict_info,file0,indent=2)

  glutPostRedisplay()

def mouse(button, state, x, y):
#  print(button,state,x,y)
  global rect_drag, mode_edit
  if mode_edit != 'bbox':
    if button == 0 and state == GLUT_DOWN:
      xy1 = my_gl.get_img_coord((x, y), img_size_info)
      for key in dict_info[mode_prsn].keys():
        if not key.startswith("keypoint_"):
          continue
        xy2 = dict_info[mode_prsn][key]
        if xy2[2] == 0:
          continue
        len = math.sqrt((xy1[0]-xy2[0])*(xy1[0]-xy2[0])+(xy1[1]-xy2[1])*(xy1[1]-xy2[1]))
        if len < 8:
          print("picked",key,len)
          mode_edit = key
      if not mode_edit == "":
        dict_info[mode_prsn][mode_edit] = [xy1[0],xy1[1],2]
  else:
    if state == GLUT_DOWN:
      dict_info[mode_prsn]['bbox'] = [x, y, 5, 5]
  glutPostRedisplay()

def motion(x, y):
  xy = my_gl.get_img_coord((x,y),img_size_info)
  if mode_edit == 'bbox' and "bbox" in dict_info[mode_prsn]:
    dict_info[mode_prsn]["bbox"][2] = abs(xy[0] - dict_info[mode_prsn]["bbox"][0])
    dict_info[mode_prsn]["bbox"][3] = abs(xy[1] - dict_info[mode_prsn]["bbox"][1])
  if not mode_edit == 'bbox' and mode_edit in dict_info[mode_prsn]:
    dict_info[mode_prsn][mode_edit][0] = xy[0]
    dict_info[mode_prsn][mode_edit][1] = xy[1]
  glutPostRedisplay()


def main():
  global list_path_json, ind_list_path_json, dict_config
  parser = argparse.ArgumentParser(description="train convolutional pose machine",
                                   add_help=True)
  parser.add_argument('--path_dir_img', type=my_util.is_dir, help='input image directory')
  parser.add_argument('--path_yml', help='input image directory')
  args = parser.parse_args()

  dict_config = yaml.load(open(args.path_yml, "r"))
  print(dict_config)

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
  ####
  list_path_json = glob.glob(args.path_dir_img+"/*.json")
  random.shuffle(list_path_json)
  list_path_json = my_dsp.arrange_old_new_json(list_path_json)
  ind_list_path_json = 0
  load_img_info(0)
  ####
  glutMainLoop()

if __name__ == "__main__":
  main()
