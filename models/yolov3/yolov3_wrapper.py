import sys
sys.path.append("/home/yantao/workspace/projects/baidu/bbox_std")

import tensorflow as tf
import os
import numpy as np
import colorsys
import logging
import json
import pickle

from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda
from models.yolov3.yolov3_model import yolo_body, yolo_eval
from models.yolov3.image_utils import letterbox_image, image_to_ndarray, letterbox_image_tf_dynamic

import pdb

class YOLOv3(object):
    _defaults = {
        "model_path": 'models/yolov3/model_data/yolov3.h5',
        "anchors_path": 'models/yolov3/model_data/yolov3_anchors.txt',
        "classes_path": 'models/yolov3/model_data/coco_classes.txt',
        "box_score_threshold": 0.3,
        "nms_iou_threshold": 0.45,
        "mAP_iou_threshold": 0.5,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name'" + n + "'"
        
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        K.set_session(self.sess)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.class_names = self._get_class()
        self.num_classes = len(self.class_names)
        self.anchors = self._get_anchors()
        self.logger.info("Loading %s model ...", self.__class__.__name__)
        self.model = self.create_model()
        self.logger.info("Model loaded.")
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(self.model.output, 
                                self.anchors, self.num_classes, 
                                self.input_image_shape,
                                score_threshold=self.box_score_threshold,
                                iou_threshold=self.nms_iou_threshold)

    def create_model(self):

        self.input_image = tf.placeholder(tf.float32, (None, None, None, 3))
        boxed_image = letterbox_image_tf_dynamic(self.input_image, (416, 416))
        input = Input(tensor=boxed_image)
        model = yolo_body(input, len(self.anchors)//3, len(self.class_names))
        
        model.load_weights(self.model_path)
        return model

    def _feed_forward(self, image):
        image_data = image_to_ndarray(image)
        image_shape = [image.size[1], image.size[0]] # Original image dimension
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],    
            feed_dict={
                self.input_image: image_data,
                self.input_image_shape: image_shape,
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes

    def predict(self, image, show_image=False):
        '''
        return dictionary of list

        Output:
        {
            'boxes' : [[top, left, bottom, right], ...]
            'scores' : [float, ...]
            'classes' : [int, ...]
        }
        '''
        
        out_boxes, out_scores, out_classes = self._feed_forward(image)
        prediction = {}
        prediction['boxes'] = []
        prediction['scores'] = []
        prediction['classes'] = []
        for temp_box, temp_score, temp_class in zip(out_boxes, out_scores, out_classes):
            prediction['boxes'].append(temp_box.tolist())
            prediction['scores'].append(temp_score)
            prediction['classes'].append(temp_class)

        return prediction
        
def main():
    image = Image.open('images/cat.jpg')
    model = YOLOv3(sess = K.get_session())

if __name__ == "__main__":
    main()