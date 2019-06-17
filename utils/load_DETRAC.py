import numpy as np 
import os
import glob
import xml.etree.ElementTree as ET

import pdb

def get_ids(dir_path):
    img_dir = os.path.join(dir_path, 'Insight-MVT_Annotation_Train')
    id_list = os.listdir(img_dir)
    id_list.sort()
    return id_list

def load_from_id(dir_path, id_name):
    img_dir = os.path.join(dir_path, 'Insight-MVT_Annotation_Train')
    det_dir = os.path.join(dir_path, 'R-CNN')
    gt_dir = os.path.join(dir_path, 'DETRAC-Train-Annotations-XML')
    imgs_path = glob.glob(os.path.join(img_dir, id_name, '*.jpg'))
    imgs_path.sort()
    num_frames = len(imgs_path)
    det_list = _load_det(os.path.join(det_dir, id_name + '_Det_R-CNN.txt'), num_frames)
    gt_list = _load_gt(os.path.join(gt_dir, id_name + '.xml'), num_frames)

    return imgs_path, det_list, gt_list

def _load_det(path, num_frames):
    """Determines the locations of the cars in the image

        Args:
            image: camera image

        Returns:
        detected objects with: bbox, confident score, class index
        [
            dictionary {
                bbox: np.array([left, up, right, down])
                score: confident_score
                class_idx: class_idx
                class_name: class name category
            }
        ]

        """

    with open(path, 'r') as f:
        lines = f.readlines()

    result_dic = {}
    for line in lines:
        line_list = line[:-1].split(',')
        frame_id = int(line_list[0])
        bbox_id = int(line_list[1])
        bbox = np.array([float(line_list[2]), float(line_list[3]), float(line_list[2]) + float(line_list[4]), float(line_list[3]) + float(line_list[5])])
        confidence_score = float(line_list[-1])

        temp_dic = {
            'bbox' : bbox,
            'score' : confidence_score,
            'class_idx' : 0,
            'class_name' : 'object',
        }

        if frame_id not in result_dic.keys():
            result_dic[frame_id] = []
        result_dic[frame_id].append(temp_dic)
    result = []
    start_idx = 1
    while start_idx <= num_frames:
        if start_idx in result_dic.keys():
            result.append(result_dic[start_idx])
        else:
            result.append([])
        start_idx += 1
    return result
        
def _load_gt(path, num_frames):
    tree = ET.parse(path)
    root = tree.getroot()
    result_dic = {}
    for frame in root.findall('frame'):
        temp_list = {}
        frame_id = int(frame.attrib['num'])
        temp_list = []
        for target in frame[0]:
            temp_dic = {}
            target_id = int(target.attrib['id'])
            bbox_dic = target.find('box').attrib
            bbox = np.array([float(bbox_dic['left']), float(bbox_dic['top']), float(bbox_dic['left']) + float(bbox_dic['width']), float(bbox_dic['top']) + float(bbox_dic['height'])])
            class_name = target.find('attribute').attrib['vehicle_type']
            temp_dic['bbox'] = bbox
            temp_dic['score'] = 1.0
            temp_dic['class_idx'] = 0
            temp_dic['class_name'] = class_name

            # used for single class detection
            temp_dic['class_name'] = 'object'

            temp_list.append(temp_dic)
        result_dic[frame_id] = temp_list
    result = []
    start_idx = 1
    while start_idx <= num_frames:
        if start_idx in result_dic.keys():
            result.append(result_dic[start_idx])
        else:
            result.append([])
        start_idx += 1
    return result
            