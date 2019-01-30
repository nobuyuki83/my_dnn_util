import cv2, argparse, shutil, json, os, math, glob
import my_dnn_util.util as my_util

def main():
  parser = argparse.ArgumentParser(description="train convolutional pose machine",
                                   add_help=True)
  parser.add_argument('--input',"-i")
  parser.add_argument('--dist',"-d", default="../pose_dataset")
  args = parser.parse_args()

  face_detector = my_util.FaceDetectorCV()

  list_path_img = glob.glob(args.input+"/*.png") + glob.glob(args.input+"/*.jpg")
  ipath_img = 0
  while True:
    print(ipath_img,len(list_path_img))
    path_img = list_path_img[ipath_img]
    frame = cv2.imread(path_img)
    ######
    frame_anno = frame.copy()
    list_rect = face_detector.get_face(frame)
    if len(list_rect) == 1:
      my_util.cv2_draw_rect(frame_anno,list_rect[0],color=(0,0,255), width=1)
    cv2.imshow("hoge",frame_anno)
    key = cv2.waitKey(-1)
    print(ipath_img,key)
    if key == 32: # ' '
      ipath_img = (ipath_img+1)%len(list_path_img)
      continue
    if key == 98: # shift ''
      ipath_img = (ipath_img+len(list_path_img)-1)%len(list_path_img)
    if key == 113:  # 'q'
      exit()
    if key == 115:
      ext = path_img.rsplit(".",1)[1]
      path_tmp = args.dist+ "/tmp."+ext
      cv2.imwrite(path_tmp, frame)
      name_md5 = my_util.md5_hash(path_tmp)
      path_trg = args.dist+"/" + name_md5 + "."+ext
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



if __name__ == "__main__":
  main()
