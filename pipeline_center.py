from tracker.kalman_filter import Tracker_center as Tracker
from utils import utils
from assign_detection_to_trackers import assign_detections_to_trackers

import cv2
import numpy as np
from PIL import Image
import copy

import pdb

def pipeline(img, det, frame_count, params_ori, is_init=False, detect_output=False, verbose=1, virtual_attack=False, return_match_info=False):
    '''
    Pipeline function for detection and tracking
    Args:
        img : nparray
            input image array
        det : object or list
            detector or detection results
        frame_count : int
            frame index
        params : dic
            parameters used for tracking
        detect_output : bool
            If True, det is detection results
        verbose : int
            verbose
        virtual_attack : bool
            If true, apply virtual attack.
    '''
    params = copy.deepcopy(params_ori)
    if detect_output:
        assert isinstance(det, list) or det == None

    tracker_list = params['tracker_list']
    max_age = params['max_age']
    min_hits = params['min_hits']
    track_id_list = params['track_id_list']
    
    frame_count += 1
    if detect_output:
        detected_objects_list = det
    else:
        detected_objects_list = det.detect_image(img)

    if virtual_attack:
        detected_objects_list = []

    if verbose == 1:
        print('Frame:', frame_count)
        print('Detected objects: ', detected_objects_list)

    x_obj = []
    img_bbox = img.copy()
    for idx, detected_object in enumerate(detected_objects_list):
        img_bbox= utils.draw_box_label(img_bbox, detected_object, box_color=(255, 0, 0), thickness=10)
    
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_obj.append(trk.obj)
    
    z_obj = [obj for obj in detected_objects_list]

    matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_obj, z_obj, iou_thrd = 0.5)  #0.3

    if verbose == 1:
        print('Detection: ', z_obj)
        print('x_obj: ', x_obj)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)
    
    # Deal with matched detections     
    if matched.size > 0:
        for trk_idx, det_idx in matched:
            z = z_obj[det_idx]['bbox']

            z_center = np.array([(z[0] + z[2]) / 2, (z[1] + z[3]) / 2])
            z_center = np.expand_dims(z_center, axis=0).T
            z_wh = np.array([z[2] - z[0] + 1, z[3] - z[1] + 1])
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.kalman_filter(z_center, z_wh)
            xx_state = tmp_trk.get_x_state().T[0].tolist()
            xx_center = [xx_state[0], xx_state[2]]
            xx_wh = tmp_trk.whRCF.get_state()

            temp_bbox = np.array([xx_center[0] - xx_wh[0] / 2, xx_center[1] - xx_wh[1] / 2, xx_center[0] + xx_wh[0] / 2, xx_center[1] + xx_wh[1] / 2]).astype('int')
            x_obj[trk_idx]['bbox'] = temp_bbox
            tmp_trk.obj['bbox'] = temp_bbox
            x_obj[trk_idx]['score'] = z_obj[det_idx]['score']
            tmp_trk.obj['score'] = z_obj[det_idx]['score']
            x_obj[trk_idx]['class_idx'] = z_obj[det_idx]['class_idx']
            tmp_trk.obj['class_idx'] = z_obj[det_idx]['class_idx']
            x_obj[trk_idx]['class_name'] = z_obj[det_idx]['class_name']
            tmp_trk.obj['class_name'] = z_obj[det_idx]['class_name']

            if not is_init:
                tmp_trk.hits += 1
            else:
                tmp_trk.hits = params_ori['min_hits']
            tmp_trk.no_losses = 0
    
    # Deal with unmatched detections      
    if len(unmatched_dets) > 0:
        for idx in unmatched_dets:
            z = z_obj[idx]['bbox']

            z_center = np.array([(z[0] + z[2]) / 2, (z[1] + z[3]) / 2])
            z_center = np.expand_dims(z_center, axis=0).T
            z_wh = np.array([z[2] - z[0] + 1, z[3] - z[1] + 1])
            tmp_trk = Tracker() # Create a new tracker
            x = np.array([[z_center[0], 0, z_center[1], 0]]).T
            tmp_trk.Init(x, z_wh)
            tmp_trk.predict_only()
            xx_state = tmp_trk.get_x_state()
            xx_state = xx_state.T[0].tolist()
            xx_center =[xx_state[0], xx_state[2]]
            xx_wh = tmp_trk.whRCF.get_state()

            temp_bbox = np.array([xx_center[0] - xx_wh[0] / 2, xx_center[1] - xx_wh[1] / 2, xx_center[0] + xx_wh[0] / 2, xx_center[1] + xx_wh[1] / 2]).astype('int')
            tmp_trk.obj['bbox'] = temp_bbox
            tmp_trk.obj['score'] = z_obj[idx]['score']
            tmp_trk.obj['class_idx'] = z_obj[idx]['class_idx']
            tmp_trk.obj['class_name'] = z_obj[idx]['class_name']

            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_obj.append(tmp_trk.obj)
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx_state = tmp_trk.get_x_state()
            xx_state = xx_state.T[0].tolist()
            xx_center =[xx_state[0], xx_state[2]]
            xx_wh = tmp_trk.whRCF.get_state()

            temp_bbox = np.array([xx_center[0] - xx_wh[0] / 2, xx_center[1] - xx_wh[1] / 2, xx_center[0] + xx_wh[0] / 2, xx_center[1] + xx_wh[1] / 2])
            tmp_trk.obj['bbox'] = temp_bbox
            x_obj[trk_idx]['bbox'] = temp_bbox
                   
    img_bbox_track = img_bbox.copy()
    # The list of tracks to be annotated  
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <= max_age)):
            good_tracker_list.append(trk)
            x_cv2 = trk.obj['bbox']
            if verbose == 1:
                print('updated box: ', x_cv2)
            img_bbox_track = utils.draw_box_label(img_bbox, trk.obj) # Draw the bounding boxes on the 

    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)  
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    if verbose == 1:
        print('Ending tracker_list: ',len(tracker_list))
        print('Ending good tracker_list: ',len(good_tracker_list))
    
    params_new = {}
    params_new['tracker_list'] = tracker_list 
    params_new['max_age'] = max_age
    params_new['min_hits'] = min_hits
    params_new['track_id_list'] = track_id_list
    
    if return_match_info:
        return img_bbox_track, params_new, (matched, unmatched_dets, unmatched_trks)
    else:
        return img_bbox_track, params_new