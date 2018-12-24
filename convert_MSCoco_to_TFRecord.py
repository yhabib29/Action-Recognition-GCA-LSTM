import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO


def load_image(filename):
    img = cv2.imread(filename)
    s = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    #img = cv2.imencode('.jpg', img)[1]
    f = open(filename, "rb")
    img = f.read()
    f.close()
    return s, img

def iou(box1, box2):
    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area
    return iou


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gen_feature1(ft_image, bbo, height, width, nb_ob):
    ft = { 'image': ft_image,
           'height': int64_feature(height),
           'width': int64_feature(width),
           'objects_number': int64_feature(nb_ob),
            'bboxes': float_feature_list(bbo)}
    return ft


# Variables
# anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
dataDir = '/home/amusaal/DATA/Coco'
dataType = 'val2014'
argv = sys.argv
if len(argv) > 1:
    dataType = argv[1]

# TFRecord output filename
train_filename = '{}/{}_yolo.tfrecords'.format(dataDir, dataType)
if os.path.isfile(train_filename):
    choice = input('file {} already exist, overwrite ? [y/n] '.format(train_filename))
    if not choice in ['y','Y']:
        sys.exit()

# Open TFREcords file
writer = tf.python_io.TFRecordWriter(train_filename)

# Annotations
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
if not os.path.isfile(annFile):
    print("Error incorrect path: " + annFile)
    sys.exit()
coco = COCO(annFile)
imgIds = coco.getImgIds()
nb_imgs = len(imgIds)

for i in range(nb_imgs):
    # Print progression
    if not i % 1000:
        print("[{}/{}]".format(str(i), str(nb_imgs)))
        sys.stdout.flush()

    # Load image
    im_id = imgIds[i]
    im = coco.loadImgs([im_id])[0]
    im_filename = im['file_name']
    imshape, image = load_image("{}/{}/{}".format(dataDir, dataType, im_filename))
    height, width, channels = imshape

    # Labels
    bbox = []
    xmin, ymin, xmax, ymax = [], [], [], []
    nb_ob = 0
    annIds = coco.getAnnIds([im_id])
    for a in coco.loadAnns(annIds):
        label = a['category_id']
        bbox.append([label] + a['bbox'])
        nb_ob += 1
    if not len(bbox) > 0:
        continue
    bbox = np.array(bbox, dtype=np.float32).flatten()
    """for bb in bbox:
        xmin.append(bb[0])
        ymin.append(bb[1])
        xmax.append(bb[2])
        ymax.append(bb[3])
    """

    # Features
    # bytes_image = tf.compat.as_bytes(image.tostring())
    bytes_image = tf.compat.as_bytes(image)
    feature_image = bytes_feature(bytes_image)

    feature = gen_feature1(feature_image, bbox, height, width, nb_ob)
    """feature = { 'image': feature_image,
                'height': int64_feature(height),
                'width': int64_feature(width),
                'objects_number': int64_feature(nb_ob),
                'class': int64_feature_list(label),
                'bboxes': float_feature_list(bbox)}"""
    """feature = { 'image': feature_image,
                'height': int64_feature(height),
                'width': int64_feature(width),
                'channels': int64_feature(channels),
                'objects_number': int64_feature(nb_ob),
                'class': int64_feature_list(label),
                'xmin': float_feature_list(xmin),
                'ymin': float_feature_list(ymin),
                'xmax': float_feature_list(xmax),
                'ymax': float_feature_list(ymax)}"""

    # Example
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()