import sys
import os
sys.path.append(os.path.abspath('./'))
from attack.attack_v2 import attack_video
from collections import deque
import numpy as np
from tqdm import tqdm
from attack.attack_v2 import cal_success_rate

if __name__ == "__main__":
    videos_info = [
            #  video file name, attack start frame,  patch coords:[left, top, right, bottom]
            ('move-in-zoomed/01.mp4', 3, [188, 284, 247, 309], None),
            ('move-in-zoomed/02.mp4', 3, [195, 233, 289, 264], None),
            ('move-in-zoomed/03.mp4', 3, [238, 194, 319, 256], None),
            ('move-in-zoomed/04.mp4', 3, [229, 231, 305, 267], None),
            ('move-in-zoomed/05.mp4', 3, [202, 172, 277, 222], None),
            ('move-in-zoomed/06.mp4', 3, [236, 230, 334, 298], None),
            ('move-in-zoomed/07.mp4', 3, [195, 203, 252, 260], None),
            ('move-in-zoomed/08.mp4', 3, [136, 193, 247, 280], None),
            ('move-in-zoomed/09.mp4', 3, [246, 210, 373, 340], None),
            ('move-in-zoomed/10.mp4', 5, [196, 205, 300, 287], None),
            ('move-out-zoomed/01.mp4', 3, [192, 213, 310, 273], [5, 0, 5, 0]),
            ('move-out-zoomed/02.mp4', 3, [143, 222, 303, 302], [5, 0, 5, 0]),
            ('move-out-zoomed/03.mp4', 3, [158, 192, 300, 283], [-5, 0, -5, 0]),
            ('move-out-zoomed/04.mp4', 3, [154, 230, 281, 289], [-5, 0, -5, 0]),
            ('move-out-zoomed/05.mp4', 3, [194, 167, 297, 249], [5, 0, 5, 0]),
            ('move-out-zoomed/06.mp4', 3, [174, 166, 326, 280], [-5, 0, -5, 0]),
            ('move-out-zoomed/07.mp4', 3, [182, 211, 304, 271], [5, 0, 5, 0]),
            ('move-out-zoomed/08.mp4', 3, [100, 131, 304, 307], [-5, 0, -5, 0]),
            ('move-out-zoomed/09.mp4', 3, [144, 188, 293, 310], [-5, 0, -5, 0]),
            ('move-out-zoomed/10.mp4', 3, [171, 159, 264, 238], [5, 0, 5, 0]),
    ]

    dir_path = './data/'
    results = []
    for idx, video_info in enumerate(tqdm(videos_info)):
        print(video_info[0])
        (video_path, temp_attack_frame, patch_bbox, moving_direction) = video_info
        video_path = os.path.join(dir_path, video_path)
        temp_attack_frame_id_list = []
        for _ in range(100):
            temp_attack_frame_id_list.append(0)
        attack_det_id_dict = {temp_attack_frame : temp_attack_frame_id_list}

        params = {
            'max_age' :  60,  #4
            'min_hits' : 6,  #1
            'tracker_list' : [],
        }
        id_list = []
        for idx in range(100):
            id_list.append(str(idx))
        params['track_id_list'] = deque(id_list)

        ret = attack_video(params, video_path=video_path, attack_det_id_dict=attack_det_id_dict, patch_bbox=patch_bbox, moving_direction=moving_direction, is_return=True)
        results.append(ret)
