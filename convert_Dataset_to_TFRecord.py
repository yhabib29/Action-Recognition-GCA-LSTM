import sys
import os
import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from pycocotools.coco import COCO


def warning(msg):
    orange = '\033[33m'
    end = '\033[0m'
    print(orange + msg + end)
    return


def error(msg):
    red = '\033[31m'
    end = '\033[0m'
    print(red + msg + end)
    sys.exit(-1)
    return


def parser():
    global dataDir, dataType
    argv = sys.argv
    argc = len(argv)
    for a, arg in enumerate(argv):
        if arg == "--dataDir" and a + 1 < argc:
            dataDir = argv[a + 1]
        elif arg == "--dataType" and a + 1 < argc:
            dataType = argv[a + 1]
    print("dataDir =\t" + dataDir)
    print("dataType =\t" + dataType)
    return


def load_image(filename):
    img = cv2.imread(filename)
    s = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # img = cv2.imencode('.jpg', img)[1]
    f = open(filename, "rb")
    img = f.read()
    f.close()
    return s, img


def load_joints(f, b, bmat):
    jts = []
    tstates = []
    for j in range(25):
        tstates.append(bmat['body'][f, b][0, 0][1][0, j][0][0][0][0][0])
        jts.append(bmat['body'][f, b][0, 0][1][0, j][0][0][5][0].tolist())
    return tstates, jts


def load_classes(ddir, dataset, fname):
    mgnd = loadmat("{}/{}_class/{}/gnd.mat".format(ddir, dataset, fname))
    gnd = [gn[0] for gn in mgnd['gnd']]
    return gnd


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_feature_list(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gen_feature1(ft_image, bbo, height, width, nb_ob):
    ft = {'image': ft_image,
          'height': int64_feature(height),
          'width': int64_feature(width),
          'objects_number': int64_feature(nb_ob),
          'bboxes': float_feature_list(bbo)}
    return ft


def gen_feature2(ft_image, joints, tstates, aclass, height, width, nb_body):
    ft = {'image': ft_image,
          'height': int64_feature(height),
          'width': int64_feature(width),
          'class': int64_feature(aclass),
          'bodies_number': int64_feature(nb_body),
          'joints': float_feature_list(joints),
          'trackingStates': float_feature_list(tstates)}
    return ft


# Generate Example for Coco Dataset
def gen_coco_example(dataDir, dataType):
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
    return


# Generate Example for Cornell Dataset
def gen_cornell_example(dataDir, dataType):
    path = "{}/{}/".format(dataDir, dataType)
    folders = os.listdir(path)

    for fid, folder in enumerate(folders):
        print('[{}/{}]'.format(str(fid), str(len(folders))), end='\r')
        sys.stdout.flush()

        if not os.path.isdir(path + folder + '/rgbjpg'):
            error('No "rgbjpg" folder in ' + path + folder)

        body_mat = loadmat(path + folder + '/body.mat')
        nb_frames, nb_body = body_mat['body'].shape
        classes = load_classes(dataDir, dataType, folder)
        for f in range(nb_frames):
            tstates_list = []
            joints_list = []
            # Load image
            img_fpath = path + folder + '/rgbjpg/' + str(f + 1).zfill(4) + '.jpg'
            if not os.path.isfile(img_fpath):
                error('No such file: ' + img_fpath)
                continue
            imshape, image = load_image(img_fpath)
            height, width, channels = imshape

            # Load joints coordinates
            for b in range(nb_body):
                isBodyTracked = body_mat['body'][f, b][0, 0][0][0][0]
                if isBodyTracked != 1:
                    continue
                trackingstates, joints = load_joints(f, b, body_mat)
                joints_list.append(joints)
                tstates_list.append(trackingstates)
            joints_list = np.array(joints_list, dtype=np.float32).flatten()
            tstates_list = np.array(tstates_list, dtype=np.float32).flatten()

            # Features
            bytes_image = tf.compat.as_bytes(image)
            feature_image = bytes_feature(bytes_image)
            feature = gen_feature2(feature_image, joints_list, tstates_list,
                                   classes[0], height, width, nb_body)

            # Example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize
            writer.write(example.SerializeToString())
    return


# Variables
# anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
dataDir = '/home/amusaal/DATA/Coco'
dataType = 'val2014'
parser()

# TFRecord output filename
suffix = "default"
if "COCO" in dataDir:
    suffix = "yolo"
elif "Cornell" in dataDir:
    suffix = "cornell"
train_filename = '{}/{}_{}.tfrecords'.format(dataDir, dataType, suffix)
if os.path.isfile(train_filename):
    choice = input('file {} already exist, overwrite ? [y/n] '.format(train_filename))
    if not choice in ['y', 'Y']:
        sys.exit()

# Open TFREcords file
writer = tf.python_io.TFRecordWriter(train_filename)

if "COCO" in dataDir:
    gen_coco_example(dataDir, dataType)
elif "Cornell" in dataDir:
    gen_cornell_example(dataDir, dataType)

writer.close()
sys.stdout.flush()
