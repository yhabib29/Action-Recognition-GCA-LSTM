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

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Variables
dataDir = '/home/amusaal/DATA/Coco'
dataType = 'val2014'
argv = sys.argv
if len(argv) > 1:
    dataType = argv[1]

# TFRecord output filename
train_filename = '{}/{}.tfrecords'.format(dataDir, dataType)
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
    label = []
    bbox = []
    xmin, ymin, xmax, ymax = [], [], [], []
    nb_ob = 0
    annIds = coco.getAnnIds([im_id])
    for a in coco.loadAnns(annIds):
        label.append(a['category_id'])
        bbox.append(a['bbox'])
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
    #feature_height = int64_feature(height)
    #feature_width = int64_feature(width)
    #feature_channels = int64_feature(channels)
    #feature_class = int64_feature_list(label)
    feature = { 'image': feature_image,
                'height': int64_feature(height),
                'width': int64_feature(width),
                'objects_number': int64_feature(nb_ob),
                'class': int64_feature_list(label),
                'bboxes': float_feature_list(bbox)}
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