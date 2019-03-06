# Research Project - F4
Activities Recognition


## YOLOv2:
Object detection

## Openpose:
Human 3D Keypoints detections

## GCA-LSTM:
Human activity recognition from Human 3D keypoints (joints).
This LTSM uses a Global Context Aware Memory cell to measure attention.
It measure the informativeness of the inputs of the second LSTM layer.
Adding Object detection in the input pipeline may improve the results.



### YOLO.py: [TRAIN ONLY/Unstable]
Tensorflow implemntation of Yolo (training only for the moment)
"For YOLOv2 we first fine tune the classification network
at the full 448 × 448 resolution for 10 epochs on ImageNet.
This gives the network time to adjust its filters to work better
on higher resolution input. We then fine tune the resulting
network on detection. This high resolution classification
network gives us an increase of almost 4% mAP."

### yolo_detector.py: [Stable]
Darknet binding - Need to install darknet

### openpose_detector.py: [Stable]
Openpose Python API example, detect Human 3D Keypoints in an image.

### convert_Dataset_to_TFRecord.py: [Stable]
Tool converting COCO and Cornell dataset to a Tensorflow friendly format (tfrecords).
