from utils import utils

from sklearn.utils.linear_assignment_ import linear_assignment
import cv2
import numpy as np
import math

import pdb

weight_same_camera = {
    'appearance' : 0.45,
    'motion' : 0.4,
    'shape' : 0.15,
    'overlap' : 0.05,
}

def assign_detections_to_trackers(trackers_obj, detections_obj, iou_thrd=0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''

    trackers = [temp_obj['bbox'] for temp_obj in trackers_obj]
    detections = [temp_obj['bbox'] for temp_obj in detections_obj]

    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    Motion_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    Shape_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t, d] = utils.box_iou(trk, det)
            Motion_mat[t, d] = get_motion_score(trk, det)
            Shape_mat[t, d] = get_shape_score(trk, det)

    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if(t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for m in matched_idx:
        if(IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(
        unmatched_detections), np.array(unmatched_trackers)


def _gaussian(x, mu, sigma):
    return math.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma))


def get_motion_score(trk, det):
    center_det = [(det[2] + det[0]) / 2, (det[3] - det[1]) / 2]
    center_trk = [(trk[2] + trk[0]) / 2, (trk[3] - trk[1]) / 2]
    width_trk = trk[2] - trk[0] + 1
    height_trk = trk[3] - trk[1] + 1
    s = _gaussian(center_trk[0], center_det[0], width_trk) * \
        _gaussian(center_trk[1], center_det[1], height_trk)
    return s

def get_shape_score(trk, det):
    width_trk = trk[2] - trk[0] + 1
    height_trk = trk[3] - trk[1] + 1
    width_det = det[2] - det[0] + 1
    height_det = det[3] - det[1] + 1

    s = (height_det - height_trk) * (width_det - width_trk) / \
        (width_det * height_det)
    return -1 * abs(s)
