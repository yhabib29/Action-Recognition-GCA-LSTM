# ProjetF4
Reconnaissance d'activités


YOLO.py:
Tensorflow implemntation of Yolo (training only for the moment)
For YOLOv2 we first fine tune the classification network
at the full 448 × 448 resolution for 10 epochs on ImageNet.
This gives the network time to adjust its filters to work better
on higher resolution input. We then fine tune the resulting
network on detection. This high resolution classification
network gives us an increase of almost 4% mAP.

yolo_detector.py:
Darknet binding

