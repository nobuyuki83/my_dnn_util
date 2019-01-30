import os, json, cv2, sys
sys.path.append('../pose_estimation')
import util

def main():
  list_keypoint_name = [
    'keypoint_nose',
    'keypoint_eyeleft',
    'keypoint_eyeright',
    'keypoint_earleft',
    'keypoint_earright',
    'keypoint_shoulderleft',
    'keypoint_shoulderright',
    'keypoint_elbowleft',
    'keypoint_elbowright',
    'keypoint_wristleft',
    'keypoint_wristright',
    'keypoint_hipleft',
    'keypoint_hipright',
    'keypoint_kneeleft',
    'keypoint_kneeright',
    'keypoint_ankleleft',
    'keypoint_ankleright' ]

#  file0 = open("annotations/person_keypoints_val2014.json")
  path_json = "/Volumes/2015_mbp_work/coco/annotations/person_keypoints_val2017.json"
  list_imginfo_anno = util.coco_get_image_annotation(path_json)
  print(len(list_imginfo_anno))
#  print(list_fname_anno)
  path_dir_out =  "/Users/nobuyuki/projects/pose_dataset"
  path_dir_img = os.path.dirname(os.path.dirname(path_json))+"/val2017"
  #####
  for imginfo_anno in list_imginfo_anno:
    dict_info0 = util.coco_get_dict_info(imginfo_anno[0],imginfo_anno[1],list_keypoint_name)
    path_img_in = path_dir_img + "/" + imginfo_anno[0]['file_name']
    path_img_out = path_dir_out + "/" + util.md5_hash(path_img_in) + "." + path_img_in.rsplit(".", 1)[1]
    if not os.path.isfile(path_img_out):
      print("already there:   ", path_img_out)
      continue
    print(path_img_in,path_img_out)
    path_json_out =  path_img_out.rsplit(".",1)[0]+".json"
    dict_info1 = json.load(open(path_json_out,"r"))
    dict_info1['person0']['segmentation'] = dict_info0['person0']['segmentation']
    if 'coco_id' in dict_info1['person0']:
      coco_id = dict_info1['person0']['coco_id']
      dict_info1['person0'].pop('coco_id')
      dict_info1['coco_id'] = coco_id
    if 'saved_time' in dict_info1:
      dict_info1.pop('saved_time')
    print(dict_info0)
    print(dict_info1)
    json.dump(dict_info1,open(path_json_out,"w"),indent=2)
    exit()

    print(path_json_out)
    np_img = cv2.imread(path_img_in)
    cv2.imshow("hoge",np_img)
    key = cv2.waitKey(-1)

  '''
      if keypoints[ip*3+2] == 1:
        cv2.circle(np_img,(x0,y0), 3, (255,0,0), -1)
      else:
        cv2.circle(np_img,(x0,y0), 3, (0,0,255), -1)
    cv2.polylines(np_img, np_sgm, True, (0, 255, 255))
    cv2.rectangle(np_img, tuple(bbox[0:2]),
                 (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                (255, 255, 255), 1)
    print(dict_out)






    print("key:",key)
    if key == 115:
      if os.path.isfile(path_img_out):
        print("already there",path_img_out)
      else:
        print(path_img_out)
        shutil.copy(path_dir_img+"/"+file_name,path_img_out)
        with open(path_dir_out+"/"+name_md5+".json","w") as file0:
          json.dump(dict_out,file0,indent=2)
    if key == 113:
      exit()
    print('\n')
  '''

if __name__ == "__main__":
  main()