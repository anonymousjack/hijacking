import numpy as np 
import cv2
import os
from PIL import Image

import pdb

def draw_box_label(img, detected_object, box_color=(0, 255, 255), thickness=4):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    bbox_cv2 = detected_object['bbox']
    bbox_cv2 = np.array(bbox_cv2).astype('int')
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, thickness)
    
    # Draw a filled box on top of the bounding box (as the background for the labels)
    cv2.rectangle(img, (left - 2, top - 45), (right + 2, top), box_color, -1, 1)

    # Output the labels that show the x and y coordinates of the bounding box center.
    text_score = str(detected_object['score'])
    cv2.putText(img, text_score, (left, top - 25), font, font_size, font_color, 1, cv2.LINE_AA)
    text_class_name = str(detected_object['class_idx'])
    cv2.putText(img, text_class_name, (left, top - 5), font, font_size, font_color, 1, cv2.LINE_AA)
    
    return img  

def box_iou(bb1, bb2):
    '''
    Calculate IoU of two bounding boxes: bb=[left, up, right, down]
    input: 
        bb1, bb2: 1*4 array or list
    output:
        scalar value
    '''
    for idx in range(4):
        bb1[idx] = float(bb1[idx])
        bb2[idx] = float(bb2[idx])
    bi = [max(bb1[0], bb2[0]), max(bb1[1], bb2[1]), min(bb1[2], bb2[2]), min(bb1[3], bb2[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        ua = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1) + (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1) - iw * ih
        iou = iw * ih / ua
    else:
        iou = 0.0

    return iou

def det4eval(det, file_id, dir_path='./det', tofile=False):
    result_list = []
    file_path = os.path.join(dir_path, file_id + '.txt')
    if tofile:
        with open(file_path, 'w') as f:
            for temp_dic in det:
                left, top, right, bottom = temp_dic['bbox'].astype(int)
                line = temp_dic['class_name'] + ' ' + str(temp_dic['score']) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + '\n'
                f.write(line)
    for temp_dic in det:
        left, top, right, bottom = temp_dic['bbox'].astype(int)
        line = temp_dic['class_name'] + ' ' + str(temp_dic['score']) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom)
        result_list.append(line)
    return result_list


def gt4eval(gt, file_id, dir_path='./gt', tofile=False):
    result_list = []
    file_path = os.path.join(dir_path, file_id + '.txt')
    if tofile:
        with open(file_path, 'w') as f:
            for temp_gt in gt:
                left, top, right, bottom = temp_gt['bbox'].astype('int')
                line = temp_gt['class_name'] + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + '\n'
                f.write(line)
    for temp_gt in gt:
        left, top, right, bottom = temp_gt['bbox'].astype('int')
        line = temp_gt['class_name'] + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom)
        result_list.append(line)
    return result_list


def trk4eval(trk, min_hits, file_id, dir_path='./trk', tofile=False):
    result_list = []
    file_path = os.path.join(dir_path, file_id + '.txt')
    if tofile:
        with open(file_path, 'w') as f:
            for temp_trk in trk:
                if temp_trk.hits < min_hits:
                    continue
                temp_obj = temp_trk.obj 
                left, top, right, bottom = temp_obj['bbox'].astype(int)
                line = temp_obj['class_name'] + ' ' + str(temp_obj['score']) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + '\n'
                f.write(line)
    for temp_trk in trk:
        if temp_trk.hits < min_hits:
            continue
        temp_obj = temp_trk.obj 
        left, top, right, bottom = temp_obj['bbox'].astype(int)
        line = temp_obj['class_name'] + ' ' + str(temp_obj['score']) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom)
        result_list.append(line)
    return result_list


def letterbox_image(
        img_np, shape=(416, 416), data_format='channels_last'):
    """Returns a letterbox image of target fname.

    Parameters
    ----------
    shape : list of integers
        The shape of the returned image (h, w).
    data_format : str
        "channels_first" or "channls_last".

    Returns
    -------
    image : array_like
        The example image.

    """
    assert len(shape) == 2
    assert data_format in ['channels_first', 'channels_last']
    image = Image.fromarray(img_np)
    iw, ih = image.size
    h, w = shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', shape, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    image = np.asarray(new_image, dtype=np.float32)
    image /= 255.
    image = image[:, :, :3]
    assert image.shape == shape + (3,)
    if data_format == 'channels_first':
        image = np.transpose(image, (2, 0, 1))
    return image, (h, w)