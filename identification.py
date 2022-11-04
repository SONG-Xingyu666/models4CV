# img_obj_det.py
# test images

from ctypes import DEFAULT_MODE
import imp
import time
import warnings
from re import L
from tqdm import tqdm
import os
from mmdet.apis import init_detector, inference_detector
import numpy as np
from utils import process_det_results, generate_obj_colors 
# from utils import display_processed_results
from mmpose.apis import inference_top_down_pose_model, inference_bottom_up_pose_model, init_pose_model, process_mmdet_results
from mmpose.datasets import DatasetInfo
import cv2
import mmcv
from mmpose.core import imshow_bboxes, imshow_keypoints


def save_img(img, img_path, output_file):
    """Save displayed image from input path to output file with the same name

    Args:
        img(ndarray): the displayed image
        img_path(str): the input path of image, to obtain the name of orginal image
        output_file(str): the output path of displayed image 
    """

    if not os.path.exists(output_file):
        os.makedirs(output_file)
    file_name = os.path.split(img_path)[1]
    save_path = os.path.join(output_file, file_name)
    cv2.imwrite(save_path,img)

def angle(a1, a2):
    """Calculate the angle between two points coordinates in angle system (0,180)

    Args:
        a1(ndarray): the coordinate of a1
        a2(ndarray): the coordinate of a2
    
    Return:
        float: the angle between a1 and a2
    """

    l1 = np.sqrt(np.dot(a1,a1))
    l2 = np.sqrt(np.dot(a2,a2))
    innerp = np.dot(a1,a2)
    cos = innerp/(l1*l2)
    angle = np.arccos(cos)
    angle = angle*180/np.pi
    return angle

def isMalicious(bbox, label, pose_results):
    
    
    if len(label) != 0:
        if label.any() == 1:
            print('malicious object detected')
            return True

    if len(pose_results) != 0:
        print('human detected')
        keypoint = pose_results[0]['keypoints']
    
        left_elbow_y = float(keypoint[7,1])
        right_elbow_y = float(keypoint[8,1])
        left_wrist_y = float(keypoint[9,1])
        right_wrist_y = float(keypoint[10,1])

        if left_wrist_y>left_elbow_y or right_wrist_y>right_elbow_y:
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
                print('malicious action detected')
                return True
                #print(angle_left, angle_right)
            else: 
                print('safe')
                return False
    else: 
        print('safe')
        return False

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
    
def process_img(img,
                det_model,
                pose_model,
                bbox_thr,
                kpt_thr,
                det_person_id,
                dataset,
                dataset_info,
                obj_colors):
    """Object detection and pose recognition inference on a single image 
       with visualization of results.

    Args:
        img(nparray): The image file path.
        det_model (nn.Module): The object detection model.
        pose_model (nn.Module): The pose estimation model.
        det_person_id (int): The id of person category in object detection dataset.
        dataset (str): The pose estimation dataset type (top-down or bottom-up).
        dataset_info (DatasetInfo): The pose estimation dataset information.
        kpt_thr (float): The keypoint score threshold.
        obj_colors (dict): The bounding box colors.
        radius (int): The keypoint radius for visualization.
        thickness (int): The link thickness for visualization.
        bbox_thr (list): The bounding box score thresholds.
        pose_nms_thr (float, optional): The OKS threshold for bottom-up pose NMS. Defaults to 0.5.
        show_score (bool, optional): Whether to show scores when visualizing. Defaults to True.

    Returns:
        ndarray: The visualized results.
    """    
    
    # object detection
    if det_model != None:
        bboxes, labels, det_results = obj_det(img=img, 
                                            det_model=det_model,
                                            bbox_thr=bbox_thr,
                                            det_person_id=det_person_id)
        
    # pose estimation
    if pose_model != None:
        pose_results = pos_est(img=img,
                                pose_model=pose_model,
                                dataset=dataset,
                                det_results=det_results,
                                det_person_id=det_person_id,
                                dataset_info=dataset_info,
                                bbox_thr=bbox_thr)

    
    
    
    # isMalicious = False
    isMal = isMalicious(bboxes, labels, pose_results)
    print(isMalicious)
    
    # visualiaztion
    img_show = display_processed_results(img,
                                        bboxes,
                                        labels,
                                        pose_results,
                                        ismalicious=isMal,
                                        kpt_thr=kpt_thr,
                                        dataset=dataset,
                                        dataset_info=None,
                                        obj_colors=obj_colors,
                                        obj_class_names=det_model.CLASSES,
                                        text_color='blue',
                                        radius=4,
                                        bbox_thickness=1,
                                        skeleton_thickness=1,
                                        text_thickness=1,
                                        show_scores=True)
    return img_show


def display_processed_results(img, 
                              bboxes,
                              labels, 
                              pose_results, 
                              ismalicious,
                              kpt_thr=0.3,
                              dataset='TopDownCocoDataset', 
                              dataset_info=None,
                              font_scale=2,
                              obj_colors=None,
                              obj_class_names=None,
                              text_color='white',
                              radius=4,
                              bbox_thickness=1,
                              skeleton_thickness=1,
                              text_thickness=1,
                              show_scores=True):
    """Draw object detection and pose estimation results over `img`.

    Args:
        img (ndarray): The image to be displayed.
        bboxes (nddarray): The object detection results after removing low scoring bboxes.
        labels (ndarray): The labels corresponding to the object detection results.
        pose_results (list): The pose estimation results.
        ismalicious (bool): if image contains malicious behavior.
        kpt_thr (float, optional): The minimum score of keypoints to be shown. 
                                   Defaults to 0.3.
        dataset (str, optional): The type of the pose estimation dataset. 
                                 Defaults to 'TopDownCocoDataset'.
        dataset_info (DatasetInfo, optional): The dataset information (containing skeletons, 
                                              links, visualization colors, etc..) 
                                              Defaults to None.
        font_scale (int, optional): The font scale for drawing texts. Defaults to 2.
        obj_colors (dict, optional): The colors of the bboxes (key-obj_class_name). 
                                     Defaults to None.
        obj_class_names (list, optional): The object class names. Defaults to None.
        text_color (str, optional): The colors of the texts. Defaults to 'white'.
        radius (int, optional): The radius for drawing keypoints. Defaults to 4.
        bbox_thickness (int, optional): The thickness for drawing bboxes. Defaults to 1.
        skeleton_thickness (int, optional): The thickness for drawing skeletons. Defaults to 1.
        text_thickness (int, optional): The thickness for drawing texts. Defaults to 1.
        show_scores (bool, optional): Whether to show the bbox scores. Defaults to True.

    Returns:
        ndarray: the displayed image.
    """

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        
        if label is None:
            continue
        
        class_name = obj_class_names[label]
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, 
                      obj_colors[class_name], thickness=bbox_thickness)
        
        # Roughly estimate the proper font size
        label_text = class_name if class_name is not None else f'class {label}'
        if len(bbox) > 4 and show_scores:
            label_text += f'{bbox[-1]:.02f}'
        text_size, text_baseline = cv2.getTextSize(label_text,
                                                    cv2.FONT_HERSHEY_DUPLEX,
                                                    font_scale, text_thickness)
        text_x1 = bbox_int[0]
        text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
        text_x2 = bbox_int[0] + text_size[0]
        text_y2 = text_y1 + text_size[1] + text_baseline
        
        cv2.putText(img, label_text, (text_x1, text_y2 - text_baseline),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    mmcv.color_val(text_color), text_thickness)
        
       
    # Visualize identification of malicious results
    if ismalicious:
        cv2.putText(img, 'Malicious', (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    else: 
        cv2.putText(img, 'Safe', (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    
    # Visualize the pose results
    pose_result = []
    for res in pose_results:
        pose_result.append(res['keypoints'])

    if dataset_info is not None:
        skeleton = dataset_info.skeleton
        pose_kpt_color = dataset_info.pose_kpt_color
        pose_link_color = dataset_info.pose_link_color
    else:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
        # TODO: These will be removed in the later versions.
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])

        if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                       'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
            # show the results
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
            ]]

        elif dataset == 'TopDownCocoWholeBodyDataset':
            # show the results
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2],
                        [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                        [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                        [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                        [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                        [102, 103], [91, 104], [104, 105], [105, 106],
                        [106, 107], [91, 108], [108, 109], [109, 110],
                        [110, 111], [112, 113], [113, 114], [114, 115],
                        [115, 116], [112, 117], [117, 118], [118, 119],
                        [119, 120], [112, 121], [121, 122], [122, 123],
                        [123, 124], [112, 125], [125, 126], [126, 127],
                        [127, 128], [112, 129], [129, 130], [130, 131],
                        [131, 132]]

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ] + [16, 16, 16, 16, 16, 16] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
            pose_kpt_color = palette[
                [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]

        elif dataset == 'TopDownAicDataset':
            skeleton = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5],
                        [8, 7], [7, 6], [6, 9], [9, 10], [10, 11], [12, 13],
                        [0, 6], [3, 9]]

            pose_link_color = palette[[
                9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 0, 7, 7
            ]]
            pose_kpt_color = palette[[
                9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 0, 0
            ]]

        elif dataset == 'TopDownMpiiDataset':
            skeleton = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                        [7, 8], [8, 9], [8, 12], [12, 11], [11, 10], [8, 13],
                        [13, 14], [14, 15]]

            pose_link_color = palette[[
                16, 16, 16, 16, 16, 16, 7, 7, 0, 9, 9, 9, 9, 9, 9
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 16, 7, 7, 0, 0, 9, 9, 9, 9, 9, 9
            ]]

        elif dataset == 'TopDownMpiiTrbDataset':
            skeleton = [[12, 13], [13, 0], [13, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [0, 6], [1, 7], [6, 7], [6, 8], [7,
                                                                 9], [8, 10],
                        [9, 11], [14, 15], [16, 17], [18, 19], [20, 21],
                        [22, 23], [24, 25], [26, 27], [28, 29], [30, 31],
                        [32, 33], [34, 35], [36, 37], [38, 39]]

            pose_link_color = palette[[16] * 14 + [19] * 13]
            pose_kpt_color = palette[[16] * 14 + [0] * 26]

        elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                         'PanopticDataset'):
            skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                        [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
                        [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                        [18, 19], [19, 20]]

            pose_link_color = palette[[
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
            pose_kpt_color = palette[[
                0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16,
                16, 16
            ]]

        elif dataset == 'InterHand2DDataset':
            skeleton = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9],
                        [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
                        [16, 17], [17, 18], [18, 19], [3, 20], [7, 20],
                        [11, 20], [15, 20], [19, 20]]

            pose_link_color = palette[[
                0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12,
                16
            ]]
            pose_kpt_color = palette[[
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16, 0
            ]]

        elif dataset == 'Face300WDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 68]
            kpt_score_thr = 0

        elif dataset == 'FaceAFLWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 19]
            kpt_score_thr = 0

        elif dataset == 'FaceCOFWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 29]
            kpt_score_thr = 0

        elif dataset == 'FaceWFLWDataset':
            # show the results
            skeleton = []

            pose_link_color = palette[[]]
            pose_kpt_color = palette[[19] * 98]
            kpt_score_thr = 0

        elif dataset == 'AnimalHorse10Dataset':
            skeleton = [[0, 1], [1, 12], [12, 16], [16, 21], [21, 17],
                        [17, 11], [11, 10], [10, 8], [8, 9], [9, 12], [2, 3],
                        [3, 4], [5, 6], [6, 7], [13, 14], [14, 15], [18, 19],
                        [19, 20]]

            pose_link_color = palette[[4] * 10 + [6] * 2 + [6] * 2 + [7] * 2 +
                                      [7] * 2]
            pose_kpt_color = palette[[
                4, 4, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 7, 7, 7, 4, 4, 7, 7, 7,
                4
            ]]

        elif dataset == 'AnimalFlyDataset':
            skeleton = [[1, 0], [2, 0], [3, 0], [4, 3], [5, 4], [7, 6], [8, 7],
                        [9, 8], [11, 10], [12, 11], [13, 12], [15, 14],
                        [16, 15], [17, 16], [19, 18], [20, 19], [21, 20],
                        [23, 22], [24, 23], [25, 24], [27, 26], [28, 27],
                        [29, 28], [30, 3], [31, 3]]

            pose_link_color = palette[[0] * 25]
            pose_kpt_color = palette[[0] * 32]

        elif dataset == 'AnimalLocustDataset':
            skeleton = [[1, 0], [2, 1], [3, 2], [4, 3], [6, 5], [7, 6], [9, 8],
                        [10, 9], [11, 10], [13, 12], [14, 13], [15, 14],
                        [17, 16], [18, 17], [19, 18], [21, 20], [22, 21],
                        [24, 23], [25, 24], [26, 25], [28, 27], [29, 28],
                        [30, 29], [32, 31], [33, 32], [34, 33]]

            pose_link_color = palette[[0] * 26]
            pose_kpt_color = palette[[0] * 35]

        elif dataset == 'AnimalZebraDataset':
            skeleton = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 7], [6, 7], [7, 2],
                        [8, 7]]

            pose_link_color = palette[[0] * 8]
            pose_kpt_color = palette[[0] * 9]

        elif dataset in 'AnimalPoseDataset':
            skeleton = [[0, 1], [0, 2], [1, 3], [0, 4], [1, 4], [4, 5], [5, 7],
                        [6, 7], [5, 8], [8, 12], [12, 16], [5, 9], [9, 13],
                        [13, 17], [6, 10], [10, 14], [14, 18], [6, 11],
                        [11, 15], [15, 19]]

            pose_link_color = palette[[0] * 20]
            pose_kpt_color = palette[[0] * 20]
        else:
            NotImplementedError()
        
        imshow_keypoints(img, pose_result, skeleton, kpt_thr,
                         pose_kpt_color, pose_link_color, radius,
                         skeleton_thickness)

    return img

def main():

# -------------identification information------------------------    
    # input and output path
    input_file = '/mnt/data/testing/asano/asano_frame/1080p/climb/climb_song_p1_1'
    output_file = '/mnt/data/results/asano/climb_song_p1_1'
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
    bbox_thr = [0.8, 0.8, 0.8] # person, launcher, cutter
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
        
        # identification
        img_show = process_img(img=img,
                    det_model=det_model,
                    pose_model=pose_model,
                    bbox_thr=bbox_thr,
                    kpt_thr=kpt_thr,
                    det_person_id=det_person_id,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    obj_colors=obj_colors)
  
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
