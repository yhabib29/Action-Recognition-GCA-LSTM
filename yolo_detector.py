import os
import sys
# Ajout du chemin du dossier où a été compilée Darknet
sys.path.insert(0, '/home/amusaal/darknetAB/python/')
import darknet
import cv2
import matplotlib.pyplot as plt



def read_class(class_file_):
    """
    Parse class file where each line is: ID CLASS_NAME
    :param class_file:                      Path to the class file
    :return: nb_classes, cids, cnames       Number of classes, Class IDs, Class names
    """
    cnames = []
    with open(class_file_, 'r') as f:
        for l in f.readlines():
            cname = l[:-1]
            cnames.append(cname)
    return cnames


def update_stats(stats, sid, score_):
    # Count
    stats[sid][0] += 1
    # Min
    if score_ < stats[sid][1]:
        stats[sid][1] = score_
    # Max
    if score_ > stats[sid][3]:
        stats[sid][3] = score_
    # Average
    avg = (stats[sid][2] * (stats[sid][0] - 1)) + score_
    stats[sid][2] = avg / stats[sid][0]
    return stats



imagePath = "/home/amusaal/DATA/kitchen.jpg"
configPath = "/home/amusaal/darknetAB/cfg/yolov2.cfg"
weightPath = "/home/amusaal/darknetAB/yolov2.weights"
metaPath= "/home/amusaal/darknetAB/cfg/coco.data"
showImage= False
makeImageOnly = False
initOnly= False
# img = cv2.imread(imagePath)
thresh = 0.25

# Initialize network
darknet.performDetect(initOnly=True)

# Detect
# darknet.performDetect(imagePath, thresh, configPath, weightPath, metaPath, showImage, makeImageOnly, initOnly):
# d2 = darknet.performDetect(imagePath='../DATA/Cornell/kitchen/data_04-22-13/rgbjpg/0001.jpg', showImage=True, makeImageOnly=True)
# d2 = darknet.performDetect(imagePath='../DATA/Cornell/kitchen/data_04-22-13/rgbjpg/0001.jpg')


dataType = 'office'
dataDir = '../DATA/Cornell/{}'.format(dataType)
classnames = read_class("/home/amusaal/darknetAB/data/coco.names")
print(len(classnames))
# class_ids = [c for c in range(1, 91) if c not in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]]
# [COUNT, MIN, AVG, MAX]
class_count = 80 * [[0, 0, 0, 0]]


folders = os.listdir(dataDir)
for i, folder in enumerate(folders):
    # Print progression
    print("[{}/{}]".format(i, len(folders)))
    sys.stdout.flush()
    files = os.listdir('{}/{}/rgbjpg/'.format(dataDir, folder))
    with open('{}_class/{}/objects.class'.format(dataDir, folder), 'w') as outfile:
        line = ''
        for file_ in files:
            fid = file_[:-4]
            detections = darknet.performDetect(imagePath='{}/{}/rgbjpg/{}'.format(dataDir, folder, file_))
            for det_ in detections:
                cid = classnames.index(det_[0])
                score = det_[1]
                bb = det_[2]
                # FRAME_ID CLASS_ID SCORE X Y W H
                line += '{} {} {} {} {} {} {}\n'.format(fid, cid, score, bb[0], bb[1], bb[2], bb[3])
                update_stats(class_count, cid, score)
        outfile.write(line)

# # Show detections
# img = cv2.imread('../DATA/Cornell/kitchen/data_04-22-13/rgbjpg/0001.jpg')
# d2 = darknet.performDetect(imageData=img.copy(), showImage=True, makeImageOnly=True)
#
# # Draw BB
# for bb in d2['detections']:
#     color = (0, 255, 0)
#     label = bb[0]
#     score = bb[1]
#     bbox = bb[2]
#     x, y = int(bbox[0]), int(bbox[1])
#     w, h = int(bbox[2]), int(bbox[3])
#     p1 = (x - w//2, y - w//2)
#     p2 = (x + w//2, y + h//2)
#     print(x,y,w,h)
#     cv2.rectangle(img, p1, p2, color, 3)
#     cv2.putText(img, label, (p1[0] - 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4)
#
# cv2.imwrite("test/img.jpg", img)
# plt.imsave("test/img2.jpg", d2['image'])


# detections = darknet.performDetect()
# detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
# print(detections)