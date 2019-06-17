# Tracker Hijacking Attack
First download the YOLOv3 weight file by:
```
wget https://perceptron-benchmark.s3-us-west-1.amazonaws.com/models/coco/yolov3.h5 -P ./models/yolov3/model_data/
```
Run the `main.py`
```
python main.py
```
The output will be the number of frames required for launching a successful tracker hijacking attack, and the position for fabricate adversarial bounding box in each attack frame. 
