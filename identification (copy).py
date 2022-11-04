# img_obj_det.py
# test images

from ctypes import DEFAULT_MODE
import time
import warnings
from re import L
from tqdm import tqdm
import os
from mmdet.apis import init_detector, inference_detector
import numpy as np
from utils import process_det_results, generate_obj_colors, display_processed_results
from mmpose.apis import inference_top_down_pose_model, inference_bottom_up_pose_model, init_pose_model, process_mmdet_results
from mmpose.datasets import DatasetInfo
import cv2


def save_img(img, img_path, output_file):
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    file_name = os.path.split(img_path)[1]
    save_path = os.path.join(output_file, file_name)
    cv2.imwrite(save_path,img)

def angle(a1, a2):
    l1 = np.sqrt(np.dot(a1,a1))
    l2 = np.sqrt(np.dot(a2,a2))
    innerp = np.dot(a1,a2)
    cos = innerp/(l1*l2)
    angle = np.arccos(cos)
    angle = angle*180/np.pi
    return angle

def isDangerObj(label):
    if 'launcher' in label:
        return True
    else: return False

def isMalicious(bbox, label, keypoint):
    
    left_elbow_y = float(keypoint[7,1])
    right_elbow_y = float(keypoint[8,1])
    left_wrist_y = float(keypoint[9, 1])
    right_wrist_y = float(keypoint[10,1])


    if 1 or 2 in label:
        res = 'malicious'
    elif left_wrist_y>left_elbow_y or right_wrist_y>right_elbow_y:
        left_hip = keypoint[11,0:2]
        right_hip = keypoint[12,0:2]
        left_knee = keypoint[13,0:2]
        right_knee = keypoint[14,0:2]
        left_ankle = keypoint[15,0:2]
        right_ankle = keypoint[16,0:2]                    
        arr1_left = left_hip - left_knee
        arr2_left = left_knee - left_ankle
        arr1_right = right_hip - right_knee
        arr2_right = right_knee - right_ankle
        angle_left = angle(arr1_left, arr2_left)
        angle_right = angle(arr1_right, arr2_right)
        if angle_right > 45 or angle_left >45:
            res = 'mailicious'
            #print(angle_left, angle_right)
        else: res = 'safe'
    else: res = 'safe'
    
    return res

def write_results(results, file):
    f = open(file, 'a')
    for result in results:
        f.writelines(str(result))
    f.close()

def get_img_list(input_file):
    img_path_list = []
    
    # if input_file is a idctionary
    if os.path.isdir(input_file):
        # Process all images in the directory.
        img_list = os.listdir(input_file)
        # Sort by file creation time.
        img_list = sorted(img_list, key=lambda x: os.path.getmtime(os.path.join(input_file, x))) if img_list else []
        for img_name in img_list:
            img_path = os.path.join(input_file, img_name)
            img_path_list.append(img_path)
    
    # if input_file is a file
    elif os.path.isfile(input_file):
        img_path_list.append(input_file) 
    
    return img_path_list

def obj_det(img, det_model, bbox_thr, det_person_id):
    # object detection inference for a single image
    det_results = inference_detector(det_model, img)
    # remove low scoring bboxes using different thresholds for different catagories respectively
    person_bbox_thr = bbox_thr[det_person_id-1] if isinstance(bbox_thr, list) else bbox_thr
    bboxes, labels  = process_det_results(det_results, bbox_thr)
    return bboxes, labels, det_results

def pos_est(img, 
            pose_model, 
            dataset, 
            det_results, 
            det_person_id, 
            dataset_info, 
            bbox_thr, 
            pose_nms_thr=0.5):
    
    # obtain person bbox threshold from bbox threshold 
    person_bbox_thr = bbox_thr[det_person_id-1] if isinstance(bbox_thr, list) else bbox_thr
    
    if dataset == 'TopDownCocoDataset':
        # Keep the person class bounding boxes.
        person_results = process_mmdet_results(det_results, det_person_id)
        # Top-down pose estimation inference for a single image.
        pose_results, _ = inference_top_down_pose_model(pose_model,
                                                        img,
                                                        person_results,
                                                        bbox_thr=person_bbox_thr,
                                                        format='xyxy',
                                                        dataset=dataset,
                                                        dataset_info=dataset_info,
                                                        return_heatmap=False,
                                                        outputs=None)
    
    elif dataset == 'BottomUpCocoDataset':
        # Bottom-up pose estimation inference for a single image.
        pose_results, _ = inference_bottom_up_pose_model(pose_model,
                                                         img,
                                                         dataset=dataset,
                                                         dataset_info=dataset_info,
                                                         pose_nms_thr=pose_nms_thr,
                                                         return_heatmap=False,
                                                         outputs=None)       
    
    return pose_results    
    

def main():

# -------------identification information------------------------    
    # input and output path
    input_file = '/mnt/data/testing/asano/asano_frame/1080p/launcher/launcher_song_p1/'
    output_file = '/mnt/data/results/asano/launcher_song_p1'
    # object detection configuration and checkpoint files 
    det_config = '/home/demachilab/mmdetection-master/configs/yolox/yolox_config.py'
    det_checkpoint = '/home/demachilab/mmdetection-master/checkpoints/yolox/yolox_s_8x8_700e_orginal3cls/best_bbox_mAP_epoch_700.pth'  
    # pose estimation configuration and cheackpoint files
    # pose_config = []
    # pose_checkpoint = []
    pose_config = '/home/demachilab/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
    pose_checkpoint = '/home/demachilab/mmpose/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    # deivice using for identification
    device = 'cuda:0'
    # person id in detection 
    det_person_id = 1 # from 1
    # threshold of detection
    bbox_thr = [0.8, 0.2, 0.2] # person, launcher, cutter
    # threshold of pose estimation
    kpt_thr = 0.3

    
    # default parameters for visualization
    # keypoint radius for visualization 
    radius = 4
    # link thickness for visualization
    thickness = 1
    # whether to show scores when visualizing
    show_score = True


    print('input file: ', input_file)
    print('output file: ', output_file)
    print('detection config file: ', det_config)
    print('detection checkpoint file: ', det_checkpoint)
    print('pose config file: ', pose_config)
    print('pose checkpoint file: ', pose_checkpoint)
    print('bbox_thr: ', bbox_thr)

# ------------------------pre-identification---------------------------
    
    # get all images path 
    img_path_list = get_img_list(input_file)
    
    # initialize models
    det_model = None
    pose_model = None
    # initialize detection model
    if len(det_config) & len(det_checkpoint):
        # initialize detector
        det_model = init_detector(det_config, det_checkpoint, device)
        # get object bounding box colors
        obj_colors = generate_obj_colors(det_model.CLASSES)
    else: 
        bboxes = []
        labels = []
        det_results = []
    # initialize estimation model
    if len(pose_config) & len(pose_checkpoint):
        # initialize pose model
        pose_model = init_pose_model(pose_config, pose_checkpoint, device)
        # Get pose estimation dataset type.
        dataset = pose_model.cfg.data['test']['type']
        # get pose estimation datasetinfo
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn('Please set `dataset_info` in the config.'
                      'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                      DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)
    else:
        pose_results = []
        dataset = []


# ------------------------------images processing------------------------------
    
    pbar = tqdm(img_path_list)
    # each image in img_list
    for img_path in pbar:
        
        # read image from img_path
        img = cv2.imread(img_path)
        
        # initialize time
        total_tic = time.time()
        # object detection
        if det_model != None:
            bboxes, labels, det_results = obj_det(img=img, 
                                               det_model=det_model,
                                               bbox_thr=bbox_thr,
                                               det_person_id=det_person_id)
            print(labels)
        # pose estimation
        if pose_model != None:
            pose_results = pos_est(img=img,
                                   pose_model=pose_model,
                                   dataset=dataset,
                                   det_results=det_results,
                                   det_person_id=det_person_id,
                                   dataset_info=dataset_info,
                                   bbox_thr=bbox_thr)

        # visualiaztion
        img_show = display_processed_results(img,
                                         bboxes,
                                         labels,
                                         pose_results,
                                         kpt_thr=kpt_thr,
                                         dataset=dataset,
                                         dataset_info=None,
                                         obj_colors=obj_colors,
                                         obj_class_names=det_model.CLASSES,
                                         text_color='blue',
                                         radius=radius,
                                         bbox_thickness=thickness,
                                         skeleton_thickness=thickness,
                                         text_thickness=thickness,
                                         show_scores=show_score)
        total_toc = time.time()
        total_time = total_toc - total_tic
        frame_rate = 1 / total_time

        
# -----------------------save results images----------------------------

        # save image
        save_img(img_show, img_path, output_file)
        # show progress
        pbar.set_description('Processing: {0} with {1:.2f} FPS'.format(img_path, frame_rate))
    
    
    


if __name__ == '__main__':
    main()
