import cv2, argparse, shutil, json, os, math
import my_dnn_util.util as my_util


def main():
  parser = argparse.ArgumentParser(description="train convolutional pose machine",
                                   add_help=True)
  parser.add_argument('--input',"-i")
  parser.add_argument('--rate',"-r", default=30)
  parser.add_argument('--dist',"-d", default="../pose_dataset")
  args = parser.parse_args()

  face_detector = my_util.FaceDetectorCV()
  video_pos = 0
  cap = cv2.VideoCapture(args.input)
  for itr in range(10000):
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_pos)
    ret, frame = cap.read()
    if not ret: break
    ######
    frame_anno = frame.copy()
    list_rect = face_detector.get_face(frame)
    if len(list_rect) == 1:
      my_util.cv2_draw_rect(frame_anno,list_rect[0],color=(0,0,255), width=1)
    cv2.imshow("hoge",frame_anno)
    key = cv2.waitKey(-1)
    print(key)
    if key == 32: # ' '
      video_pos += int(args.rate)
    if key == 102: # 'f'
      video_pos += int(args.rate)*3
    if key == 113:  # 'q'
      exit()
    if key == 115:
      path_tmp = args.dist+ "/tmp.png"
      cv2.imwrite(path_tmp, frame)
      name_md5 = my_util.md5_hash(path_tmp)
      path_trg = args.dist+"/" + name_md5 + ".png"
      if os.path.isfile(path_trg):
        print("already there", path_trg)
      else:
        shutil.move(path_tmp, path_trg)
        dict_info = {}
        dict_info["person0"] = {}
        dict_info["person0"]["face_rad"] = 80
        if len(list_rect) == 1:
          rw = list_rect[0][2]
          rh = list_rect[0][3]
          dict_info["person0"]["face_rad"] = math.sqrt(rw*rw+rh*rh)*0.5
          dict_info["person0"]["keypoint_head"] = [list_rect[0][0]+rw*0.5,list_rect[0][1]+rh*0.5,2]
        with open(args.dist+"/" + name_md5 + ".json", "w") as file0:
          json.dump(dict_info, file0, indent=2)

  cap.release()


if __name__ == "__main__":
  main()
