# Research Project - F4
Activities Recognition

----------------

## YOLOv2:
Object detection [3]

https://github.com/AlexeyAB/darknet

## Openpose:
Human 3D Keypoints detections

https://github.com/CMU-Perceptual-Computing-Lab/openpose

## GCA-LSTM:
Human activity recognition from Human 3D keypoints (joints). [1] [2]
This LTSM uses a Global Context Aware Memory cell to measure attention.
It measure the informativeness of the inputs of the second LSTM layer.
Adding Object detection in the input pipeline may improve the results.

------------------

### GCA-LSTM.py [Stable]
Main file for activity recognition.
Using Cornell Dataset [4]

### ST-LSTM.py [Stable]
Implementation of a SpatioTemporal LSTM Cell [1] [2] based on Tensorflow source code and other references:
- add references

### YOLO.py: [TRAIN ONLY/Unstable]
Tensorflow implemntation of Yolo (training only for the moment)
"For YOLOv2 we first fine tune the classification network
at the full 448 × 448 resolution for 10 epochs on ImageNet.
This gives the network time to adjust its filters to work better
on higher resolution input. We then fine tune the resulting
network on detection. This high resolution classification
network gives us an increase of almost 4% mAP."

Problems: MEMORY LEAK - load all the dataset in the GPU memory

### yolo_detector.py: [Stable]
Darknet binding - Need to install darknet

### openpose_detector.py: [Stable]
Openpose Python API example, detect Human 3D Keypoints in an image.

### convert_Dataset_to_TFRecord.py: [Stable]
Script converting _COCO_ and _Cornell_ [4] datasets to a Tensorflow friendly format (tfrecords).

--------------------

### References

[1] "_Skeleton Based Human Action Recognition with Global Context-Aware Attention LSTM Networks_", Jun Liu, Gang Wang, Ling-Yu Duan, Kamila Abdiyeva, and Alex C. Kot

[2] "_Skeleton-Based Action Recognition Using Spatio-Temporal LSTM Network with Trust Gates_", Jun Liu, Amir Shahroudy, Dong Xu, and Gang Wang

[3] "_YOLO9000: Better, Faster, Stronger_", Joseph Redmon, Ali Farhadi

[4] http://watchnpatch.cs.cornell.edu/
