from OpenGL.GL import *
import OpenGL.GLUT as glut
import cv2, math

def pick_loop_vertex(xy1:list,key:str,dict_prsn:dict):
  iloop_selected = -1
  ivtx_selected = -1
  if not key in dict_prsn:
    return
  min_len = -1.0
  for iloop, loop in enumerate(dict_prsn[key]):
    for ivtx in range(len(loop) // 2):
      xy0 = loop[ivtx * 2 + 0], loop[ivtx * 2 + 1]
      len0 = math.sqrt((xy0[0] - xy1[0]) * (xy0[0] - xy1[0]) + (xy0[1] - xy1[1]) * (xy0[1] - xy1[1]))
      if min_len < 0 or len0 < min_len:
        min_len = len0
        iloop_selected = iloop
        ivtx_selected = ivtx
  return iloop_selected,ivtx_selected

def pick_loop_edge(xy2:list, key:str, dict_prsn:dict):
  iloop_selected = -1
  ivtx_selected = -1
  if not key in dict_prsn:
    return iloop_selected,ivtx_selected, [0,0]
  min_len = -1.0
  min_xy3 = [0,0]
  for iloop, loop in enumerate(dict_prsn[key]):
    for iv0 in range(len(loop) // 2):
      iv1 = (iv0+1)%(len(loop)//2)
      xy0 = loop[iv0 * 2 + 0], loop[iv0 * 2 + 1]
      xy1 = loop[iv1 * 2 + 0], loop[iv1 * 2 + 1]
      v01 = xy1[0]-xy0[0],xy1[1]-xy0[1]
      v02 = xy2[0]-xy0[0],xy2[1]-xy0[1]
      ratio_selected = (v01[0]*v02[0]+v01[1]*v02[1])/(v01[0]*v01[0]+v01[1]*v01[1])
      if ratio_selected < 0.2: ratio_selected = 0.2
      if ratio_selected > 0.8: ratio_selected = 0.8
      xy3 = (1-ratio_selected)*xy0[0]+ratio_selected*xy1[0],(1-ratio_selected)*xy0[1]+ratio_selected*xy1[1]
      len23 = math.sqrt((xy2[0] - xy3[0]) * (xy2[0] - xy3[0]) + (xy2[1] - xy3[1]) * (xy2[1] - xy3[1]))
      if min_len < 0 or len23 < min_len:
        min_len = len23
        min_xy3 = xy3
        iloop_selected = iloop
        ivtx_selected = iv1
  return iloop_selected,ivtx_selected,min_xy3


def area_loop(loop):
  nv = len(loop)//2
  area = 0.0
  for iv0 in range(nv):
    iv1 = (iv0+1)%nv
    xy0 = loop[iv0 * 2 + 0], loop[iv0 * 2 + 1]
    xy1 = loop[iv1 * 2 + 0], loop[iv1 * 2 + 1]
    area += xy0[0]*xy1[1] - xy0[1]*xy1[0]
  return 0.5*area


########################################################################################3
def load_texture(img_bgr):
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  width_org = img_rgb.shape[1]
  height_org = img_rgb.shape[0]
  width_po2 = int(math.pow(2, math.ceil(math.log(width_org, 2))))
  height_po2 = int(math.pow(2, math.ceil(math.log(height_org, 2))))
  img_rgb = cv2.copyMakeBorder(img_rgb,
                               0, height_po2 - height_org, 0, width_po2 - width_org,
                               cv2.BORDER_CONSTANT, (0, 0, 0))
  rw = width_org / width_po2
  rh = height_org / height_po2
  glEnable(GL_TEXTURE_2D)
  texid = glGenTextures(1)
  glBindTexture(GL_TEXTURE_2D, texid)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_po2, height_po2, 0, GL_RGB, GL_UNSIGNED_BYTE, img_rgb)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
  return (width_org, height_org, rw, rh, texid)

def drawRect(rect, color=(1, 0, 0), width=1):
  plt = (rect[0], -rect[1])
  size_rect_w = rect[2]
  size_rect_h = rect[3]
  glDisable(GL_LIGHTING)
  glDisable(GL_TEXTURE_2D)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINE_LOOP)
  glVertex2f(plt[0], plt[1])
  glVertex2f(plt[0] + size_rect_w, plt[1])
  glVertex2f(plt[0] + size_rect_w, plt[1] - size_rect_h)
  glVertex2f(plt[0], plt[1] - size_rect_h)
  glEnd()

def drawCircle(cnt, rad, color=(1, 0, 0), width=1):
  glDisable(GL_TEXTURE_2D)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINE_LOOP)
  ndiv = 32
  dt = 3.1415*2.0/ndiv
  for i in range(32):
    glVertex3f(+cnt[0]+rad*math.cos(dt*i),
               -cnt[1]+rad*math.sin(dt*i), -0.1)
  glEnd()
  ###


def drawLine(cnt0, cnt1, color=(1, 0, 0), width=1):
  glDisable(GL_TEXTURE_2D)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINES)
  glVertex3f(+cnt0[0],-cnt0[1], -0.1)
  glVertex3f(+cnt1[0],-cnt1[1], -0.1)
  glEnd()


def drawPolyline(pl, color=(1, 0, 0), width=1):
  glDisable(GL_TEXTURE_2D)
  glDisable(GL_LIGHTING)
  glColor3d(color[0], color[1], color[2])
  glLineWidth(width)
  glBegin(GL_LINE_LOOP)
  for ipl in range(len(pl)//2):
    glVertex2f(pl[ipl*2+0], -pl[ipl*2+1])
  glEnd()



def set_view_trans(img_size_info):
  viewport = glGetIntegerv(GL_VIEWPORT)
  win_h = viewport[3]
  win_w = viewport[2]
  img_w = img_size_info[0]
  img_h = img_size_info[1]
  #####
  scale_imgwin = max(img_h / win_h, img_w / win_w)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glOrtho(0, win_w * scale_imgwin, -win_h * scale_imgwin, 0, -1000, 1000)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()

def draw_img(img_size_info):
  img_w = img_size_info[0]
  img_h = img_size_info[1]
  imgtex_w = img_size_info[2]
  imgtex_h = img_size_info[3]
  ####
  glDisable(GL_LIGHTING)
  glEnable(GL_TEXTURE_2D)
  id_tex_org = img_size_info[4]
  if id_tex_org is not None and glIsTexture(id_tex_org):
    glBindTexture(GL_TEXTURE_2D, id_tex_org)
  glColor3d(1, 1, 1)
  glBegin(GL_QUADS)
  ## left bottom
  glTexCoord2f(0.0, imgtex_h)
  glVertex2f(0, -img_h)
  ## right bottom
  glTexCoord2f(imgtex_w, imgtex_h)
  glVertex2f(img_w, -img_h)
  ### right top
  glTexCoord2f(imgtex_w, 0.0)
  glVertex2f(img_w, 0)
  ## left top
  glTexCoord2f(0.0, 0.0)
  glVertex2f(0, 0)
  glEnd()



def get_img_coord(xy, img_size_info):
  ####
  viewport = glGetIntegerv(GL_VIEWPORT)
  win_h = viewport[3]
  win_w = viewport[2]
  img_w = img_size_info[0]
  img_h = img_size_info[1]
#  print(win_h,win_w,img_h,img_w)
  ####
  scale_imgwin = max(img_h / win_h, img_w / win_w)
  x1 = xy[0] * scale_imgwin
  y1 = xy[1] * scale_imgwin
  return (x1, y1)

def draw_segmentation_loops(dict_info,selected_loop:int,name_seg:str):
  if name_seg in dict_info:
    for iloop,loop in enumerate(dict_info[name_seg]):
      drawPolyline(loop,color=(1,1,1),width=1)
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



#####################################################################################
# glut dataset from here


def glut_print( x,  y,  font,  text, color):
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glColor3f(color[0],color[1],color[2])
  glRasterPos2f(x,y)
  for ch in text:
    glut.glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )

def glut_edit_segmentationloops(state,button,xy1,name_seg,dict_info):
  if state == glut.GLUT_DOWN:
    if button == glut.GLUT_LEFT_BUTTON:
      iloop_selected, ivtx_selected = pick_loop_vertex(xy1, name_seg, dict_info)
    if button == glut.GLUT_MIDDLE_BUTTON:
      iloop_selected, ivtx_selected = pick_loop_vertex(xy1, name_seg, dict_info)
      if not ivtx_selected == -1:
        loop = dict_info[name_seg][iloop_selected]
        loop.pop(ivtx_selected * 2 + 1)
        loop.pop(ivtx_selected * 2 + 0)
        ivtx_selected = -1
    elif button == glut.GLUT_RIGHT_BUTTON:
      iloop_selected, ivtx_selected, xy3 = pick_loop_edge(xy1, name_seg, dict_info)
      if not ivtx_selected == -1:
        loop = dict_info[name_seg][iloop_selected]
        loop.insert(ivtx_selected * 2 + 0, xy1[0])
        loop.insert(ivtx_selected * 2 + 1, xy1[1])
  elif state == glut.GLUT_UP:
    ivtx_selected = -1
    iloop_selected = -1
  return iloop_selected,ivtx_selected

def glut_move_segmentationloop(button,name_seg,iloop_selected,ivtx_selected,dict_info,xy):
  if button == glut.GLUT_LEFT_BUTTON:
    if name_seg in dict_info:
      if iloop_selected >= 0 and iloop_selected < len(dict_info[name_seg]):
        loop = dict_info[name_seg][iloop_selected]
        if ivtx_selected >= 0 and ivtx_selected * 2 < len(loop):
          loop[ivtx_selected * 2 + 0] = xy[0]
          loop[ivtx_selected * 2 + 1] = xy[1]

def glut_add_point_segmentation_loops(state,name_seg,dict_info,xy1):
  if state == glut.GLUT_DOWN:
    if not name_seg in dict_info:
      dict_info[name_seg] = [[]]
    iloop = len(dict_info[name_seg]) - 1
    dict_info[name_seg][iloop].extend([int(xy1[0]), int(xy1[1])])

def clean_loop(name_seg,dict_info):
  if name_seg in dict_info:
    list_loop = dict_info[name_seg]
    list_loop_new = []
    for loop in list_loop:
      if len(loop) == 0:
        print("deleting new loop")
      else:
        list_loop_new.append(loop)
    dict_info[name_seg] = list_loop_new