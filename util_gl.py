from OpenGL.GL import *
from OpenGL.GLUT import *
import cv2, math

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

def glut_print( x,  y,  font,  text, color):
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  glColor3f(color[0],color[1],color[2])
  glRasterPos2f(x,y)
  for ch in text:
    glutBitmapCharacter( font , ctypes.c_int( ord(ch) ) )

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


#####################################################################################
# person dataset from here

