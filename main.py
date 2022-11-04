import argparse
import time
import cv2
import os
import warnings
from tqdm import tqdm

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, inference_bottom_up_pose_model,
                         init_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo
from utils import generate_obj_colors, process_det_results, display_processed_results


def parse_args():
    parser = argparse.ArgumentParser(description='Image processing.')
    parser.add_argument('det_config', 
                        help='Config file for object detection model.')
    parser.add_argument('det_checkpoint', 
                        help='Checkpoint file for object detection model.')
    parser.add_argument('pose_config', 
                        help='Config file for pose estimation model.')
    parser.add_argument('pose_checkpoint', 
                        help='Checkpoint file for pose estimation model.')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='CPU/CUDA device option.')
    parser.add_argument('--det-person-id', type=int, default=1, 
                        help='Person id for bounding box detection model.')
    parser.add_argument('--bbox_thr', type=float, default=[0.8, 0.4, 0.4], nargs=3, 
                        help='Bbox score thresholds.')
    parser.add_argument('--kpt-thr', type=float, default=0.3, 
                        help='Keypoint score threshold.')
    parser.add_argument('--pose-nms-thr', type=float,default=0.5,
                        help='OKS threshold for bottom-up pose NMS.')
    parser.add_argument('--radius', type=int, default=4,
                        help='Keypoint radius for visualization.')
    parser.add_argument('--thickness', type=int, default=1,
                        help='Link thickness for visualization.')
    parser.add_argument('--img-src', type=str, 
                        help='Directory of images/name of a image.')
    parser.add_argument('--save-dir', type=str, 
                        help='Save directory of output images.')
    parser.add_argument('--show-score', action='store_true',
                        help='Whether to show scores when visualizing.')
    args = parser.parse_args()
    return args

def process_img(img_path, 
                det_model, 
                pose_model, 
                det_person_id,
                dataset,
                dataset_info, 
                kpt_thr, 
                obj_colors, 
                radius, 
                thickness, 
                bbox_thr,
                pose_nms_thr=0.5,
                show_score=True):
    """Object detection and pose recognition inference on a single image 
       with visualization of results.

    Args:
        img_path (str): The image file path.
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
        float: The frame rate.
    """    
    
    total_tic = time.time()

    img = cv2.imread(img_path)

    # Object detection inference for a single image.
    mmdet_results = inference_detector(det_model, img)

    # Remove low scoring bboxes using different thresholds for different categories respectively.
    person_bbox_thr = bbox_thr[det_person_id-1] if isinstance(bbox_thr, list) else bbox_thr
    bboxes, labels  = process_det_results(mmdet_results, person_bbox_thr)
    
    if dataset == 'TopDownCocoDataset':
        # Keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_person_id)

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
    
    return img_show, frame_rate

def main():
    args = parse_args()

    # Build the detection model from a config file and a checkpoint file.
    det_model = init_detector(args.det_config, args.det_checkpoint, 
                              device=args.device.lower())

    # Get object bounding box colors.
    obj_colors = generate_obj_colors(det_model.CLASSES)

    # Build the pose model from a config file and a checkpoint file.
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, 
                                 device=args.device.lower())

    # Get pose estimation dataset type.
    dataset = pose_model.cfg.data['test']['type']
    
    # Get pose estimation datasetinfo.
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn('Please set `dataset_info` in the config.'
                      'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                      DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    if os.path.isdir(args.img_src):
        # Process all images in the directory.
        img_list = os.listdir(args.img_src)
        # Sort by file creation time.
        img_list = sorted(img_list, 
                          key=lambda x: os.path.getmtime(os.path.join(args.img_src, x))) if img_list else []
        pbar = tqdm(img_list)
        for img_name in pbar:
            img_path = os.path.join(args.img_src, img_name)   
            img_show, frame_rate = process_img(img_path,
                                               det_model,
                                               pose_model,
                                               args.det_person_id,
                                               dataset,
                                               dataset_info,
                                               args.kpt_thr,
                                               obj_colors,
                                               args.radius,
                                               args.thickness,
                                               args.bbox_thr,
                                               args.show_score)
            
            if args.save_dir:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                file_name = os.path.split(img_path)[1]
                save_path = os.path.join(args.save_dir, file_name)
                cv2.imwrite(save_path, img_show)
            else:
                cv2.imshow('image', img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            
            pbar.set_description('Processing: {0} with {1:.2f} FPS'.format(img_path, frame_rate))

    elif os.path.isfile(args.img_src):
        # Processing a single image file.  
        img_show, frame_rate = process_img(args.img_src,
                                           det_model,
                                           pose_model,
                                           args.det_person_id,
                                           dataset,
                                           dataset_info,
                                           args.kpt_thr,
                                           obj_colors,
                                           args.radius,
                                           args.thickness,
                                           args.bbox_thr,
                                           args.show_score)

        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            file_name = os.path.split(args.img_src)[1]
            save_path = os.path.join(args.save_dir, file_name)
            cv2.imwrite(save_path, img_show)
        else:
            cv2.imshow('image', img_show)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
            
        print('Processing: {0} with {1:.2f} FPS'.format(args.img_src, frame_rate))

if __name__ == '__main__':
    main()