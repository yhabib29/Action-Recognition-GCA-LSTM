import sys
sys.path.insert(0, '/home/amusaal/darknetAB/python/')
import darknet
import cv2


imagePath = "/home/amusaal/DATA/kitchen.jpg"
configPath = "/home/amusaal/darknetAB/cfg/yolov2.cfg"
weightPath = "/home/amusaal/darknetAB/yolov2.weights"
metaPath= "/home/amusaal/darknetAB/cfg/coco.data"
showImage= False
makeImageOnly = False
initOnly= False
img = cv2.imread(imagePath)
thresh = 0.25

# Initialize network
darknet.performDetect(initOnly=True)

# Detect
# darknet.performDetect(imagePath, thresh, configPath, weightPath, metaPath, showImage, makeImageOnly, initOnly):
d1 = darknet.performDetect()
d2 = darknet.performDetect(imagePath=imagePath)

print(d1)
print(d2)

# detections = darknet.performDetect()
# detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
# print(detections)