import warnings
import numpy as np
from numpy import random
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import mmcv
from mmpose.core import imshow_bboxes, imshow_keypoints

import colorsys
import random
 
def generate_hls_colors(len_classes):
    """Randomly generate hls colors corresponding to the number of object categories.

    Args:
        len_classes (int): The number of object categories.

    Returns:
        list: The generated hls colors.
    """    

    return [[i / 360.0, 
             (50 + random.random() * 10) / 100.0, 
             (90 + random.random() * 10) / 100.0] 
            for i in range(0, 360, int(360.0 / len_classes))]
 
def generate_rgb_colors(len_classes):
    """Randomly generate rgb colors corresponding to the number of object categories.

    Args:
        len_classes (int): The number of object categories.

    Returns:
        list: The generated rgb colors.
    """   

    rgb_colors = [[int(i * 255.0) for i in colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])]
                  for hlsc in generate_hls_colors(len_classes)]
    
    return [] if len_classes < 1 else rgb_colors

def generate_obj_colors(obj_class_names):
    """Randomly generate different colors for bboxes of each category of objects.

    Args:
        obj_class_names (list): The names of object categories.

    Returns:
        dict: The generated bboxes colors.
    """    
    
    #return {obj_class_name: [random.randint(0, 255) for _ in range(3)] for obj_class_name in obj_class_names}
    rgb_colors = generate_rgb_colors(len(obj_class_names))
    return {obj_class_name: rgb_colors[i] for i, obj_class_name in enumerate(obj_class_names)}

def display_results(img, 
                    det_results, 
                    pose_results, 
                    bbox_thr=0.3,
                    kpt_thr=0.3,
                    dataset='TopDownCocoDataset', 
                    dataset_info=None,
                    font_scale=0.5,
                    obj_colors=None,
                    obj_class_names=None,
                    text_color='white',
                    radius=4,
                    thickness=1):
    if isinstance(det_results, tuple):
        bbox_result, segm_result = det_results
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = det_results, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    if bbox_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > bbox_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if label is None:
            continue
        class_name = obj_class_names[label]
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, obj_colors[class_name], thickness=thickness)
        # roughly estimate the proper font size
        label_text = class_name if class_name is not None else f'class {label}'
        if len(bbox) > 4:
            label_text += f'{bbox[-1]:.02f}'
        text_size, text_baseline = cv2.getTextSize(label_text,
                                                    cv2.FONT_HERSHEY_DUPLEX,
                                                    font_scale, thickness)
        text_x1 = bbox_int[0]
        text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
        text_x2 = bbox_int[0] + text_size[0]
        text_y2 = text_y1 + text_size[1] + text_baseline
        #cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color,
        #                cv2.FILLED)
        cv2.putText(img, label_text, (text_x1, text_y2 - text_baseline),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    mmcv.color_val(text_color), thickness)
    
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
                         thickness)

    return img

def process_det_results(det_results, bbox_thr):
    """Removing low scoring bboxes using different thresholds 
       for different categories respectively.

    Args:
        det_results (list): The object detection results.
        bbox_thr (list or float): The thresholds for each category to remove bboxes 
                                  (or use a uniform threshold for different categories 
                                   if it is a float type).

    Returns:
        ndarray: The detection results after removing low scoring bboxes.
        ndarray: The labels corresponding to the detection results.
    """    
    
    bboxes = np.vstack(det_results)
    labels = [np.full(bbox.shape[0], i, dtype=np.int32)
              for i, bbox in enumerate(det_results)]
    labels = np.concatenate(labels)

    if isinstance(bbox_thr, list):
        bbox_thr = np.array(bbox_thr)[labels]

    assert bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > bbox_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]

    return bboxes, labels

def display_processed_results(img, 
                              bboxes,
                              labels, 
                              pose_results, 
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
        
        # if label == 1 or 2:
        #     cv2.putText(img, 'Malicious', (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # else: 
        #     cv2.putText(img, 'Safe', (5,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
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