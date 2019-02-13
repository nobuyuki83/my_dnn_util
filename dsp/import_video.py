import cv2, argparse, shutil, json, os, math, numpy
import my_dnn_util.util as my_util

def save(path_dir_dist, np_img_in, face_pos_rad):
  path_tmp = path_dir_dist + "/tmp.png"
  cv2.imwrite(path_tmp, np_img_in)
  name_md5 = my_util.md5_hash(path_tmp)
  path_trg = path_dir_dist + "/" + name_md5 + ".png"
  if os.path.isfile(path_trg):
    print("already there", path_trg)
  else:
    shutil.move(path_tmp, path_trg)
    dict_info = {}
    dict_info["person0"] = {}
    dict_info["person0"]["rad_head"] = 80
    if len(face_pos_rad) == 3:
      dict_info["person0"]["rad_head"] = face_pos_rad[2]
      dict_info["person0"]["kp_head"] = [face_pos_rad[0], face_pos_rad[1], 2]
    with open(path_dir_dist + "/" + name_md5 + ".json", "w") as file0:
      json.dump(dict_info, file0, indent=2)

def main():
  parser = argparse.ArgumentParser(description="train convolutional pose machine",
                                   add_help=True)
  parser.add_argument('--path_video',"-i", required=True)
  parser.add_argument('--rate',"-r", default=30)
  parser.add_argument('--rotate', default=0)
  parser.add_argument('--scale', default=1)
  parser.add_argument('--path_dir_dist',"-d", required=True)
  args = parser.parse_args()

  face_detector = my_util.FaceDetectorCV()
  video_pos = 0
  cap = cv2.VideoCapture(args.path_video)
  for itr in range(10000):
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_pos)
    ret, frame = cap.read()
    if not ret: break
    #####
    frame = numpy.rot90(frame,int(args.rotate))
    frame = frame[::int(args.scale),::int(args.scale),:]
    ######
    frame_anno = frame.copy()
    list_rect = face_detector.get_face(frame)
    face_pos_rad = []
    if len(list_rect) == 1:
      my_util.cv2_draw_rect(frame_anno,list_rect[0],color=(0,0,255), width=1)
      rw = list_rect[0][2]
      rh = list_rect[0][3]
      face_pos_rad = [
        list_rect[0][0] + rw * 0.5,
        list_rect[0][1] + rh * 0.5,
        math.sqrt(rw * rw + rh * rh) * 0.5 ]
    cv2.imshow("hoge",frame_anno)
    key = cv2.waitKey(-1)
    print(key)
    if key == 32: # ' '
      video_pos += int(args.rate)
    if key == 102: # 'f'
      video_pos += int(args.rate)*4
    if key == 103: # 'g'
      video_pos += int(args.rate)*12
    if key == 113:  # 'q'
      exit()
    if key == 115: # 's'
      print(face_pos_rad)
      save(args.path_dir_dist,frame,face_pos_rad)
  cap.release()


if __name__ == "__main__":
  main()
