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


def check_filename(fname):
    # TFRecord output filename
    if os.path.isfile(fname):
        choice = input('file {} already exist, overwrite ? [y/n] '.format(fname))
        if not choice in ['y', 'Y']:
            sys.exit(0)
    return


def get_help():
    gbold = '\033[1;32m'
    green = '\033[0;32m'
    dpath_1 = '../home/amusaal/DATA/Coco'
    dpath_2 = '../home/amusaal/DATA/Cornell'
    dtype_1 = 'val2014'
    dtype_2 = 'kitchen'
    mhelp =  gbold + "This script is used to convert Coco and Cornell datasets to TFRecords file\n"
    mhelp += "COCO:\t\tdataDir is the directory containing the folder dataType which contains all images\n"
    mhelp += "\t\tCOCO Python API is required (pip install pycocotools)\n"
    mhelp += "Cornell:\tdataDir is the directory containing the folder dataType (kitchen/office)\n"
    mhelp += "\t\tIt also contains split files, and classnames files.\n\n"
    mhelp += "--help [-h]\t\t\t" + green
    mhelp += "Show help\n"
    mhelp += gbold + "--dataDir [-d]\tPATH\t\t" + green
    mhelp += "Path to the directory of the dataset\n"
    mhelp += gbold + "--dataType [-t]\tNAME\t\t" + green
    mhelp += "Name of the folder containing data. It will be used as the output prefix.\n\n"
    mhelp += gbold + "Example:\n" + green
    mhelp += "python3 convert_Dataset_to_TFRecord.py --dataDir {} --dataType {}\n".format(dpath_1, dtype_1)
    mhelp += "python3 convert_Dataset_to_TFRecord.py -d {} -t {}\n\n".format(dpath_2, dtype_2)
    mhelp += '\033[0m'
    return mhelp


def parser():
    global dataDir, dataType
    argv = sys.argv
    argc = len(argv)
    help_ = get_help()
    for a, arg in enumerate(argv):
        if a == argc-1:
            if arg in ['--help', '-h']:
                print(help_)
                sys.exit()
            break
        elif arg in ['--help', '-h']:
            print(help_)
            sys.exit()
        elif arg in ["--dataDir", "-d"]:
            if os.path.isdir(argv[a + 1]):
                dataDir = argv[a + 1]
            else:
                error('Error: Invalid directory {}'.format(argv[a + 1]))
        elif arg in ["--dataType", "-t"]:
            dataType = argv[a + 1]
        else:
            continue
    if not os.path.isdir('{}/{}'.format(dataDir, dataType)):
        error('Error: could not find {} in {}'. format(dataType,dataDir))
    print("dataDir =\t" + dataDir)
    print("dataType =\t" + dataType)
    return


def load_image(filename):
    img = cv2.imread(filename)
    s = img.shape
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32)
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


def gen_feature1(ft_image, height, width, nb_ob):
    ft = {'image': ft_image,
          'height': int64_feature(height),
          'width': int64_feature(width),
          'objects_number': int64_feature(nb_ob)}
    return ft


def gen_feature2(ft_image, joints, tstates, aclass, bodies_id):
    ft = {'image': ft_image,
          'class': int64_feature(aclass),
          'bodies': int64_feature_list(bodies_id),
          'joints': float_feature_list(joints),
          'trackingStates': int64_feature_list(tstates)}
    return ft


# Generate Example for Coco Dataset
def gen_coco_example(dataDir, dataType):
    # Open TFRecords file
    filename = '{}/{}_coco.tfrecords'.format(dataDir, dataType)
    check_filename(filename)
    writer = tf.python_io.TFRecordWriter(filename)
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
        nb_ob = 0
        annIds = coco.getAnnIds([im_id])
        for a in coco.loadAnns(annIds):
            label = a['category_id']
            bb = np.array(a['bbox']) / np.array((width,height,width,height))
            bbox.append([label] + bb.tolist())
            nb_ob += 1
        if not len(bbox) > 0:
            continue
        bbox = np.array(bbox, dtype=np.float32)#.flatten()

        # Features
        # bytes_image = tf.compat.as_bytes(image.tostring())
        bytes_image = tf.compat.as_bytes(image)
        feature_image = bytes_feature(bytes_image)

        feature = gen_feature1(feature_image, height, width, nb_ob)

        # Feature list
        context = tf.train.Features(feature=feature)
        fl_bboxes = tf.train.FeatureList(feature=[float_feature_list(bbo) for bbo in bbox])
        feature_list = tf.train.FeatureLists(feature_list={'bboxes': fl_bboxes})

        # Example
        # example = tf.train.Example(features=tf.train.Features(feature=feature))
        example = tf.train.SequenceExample(context=context, feature_lists=feature_list)
        # Serialize
        writer.write(example.SerializeToString())
    # Close writer
    writer.close()
    return


# Generate Example for Cornell Dataset
def gen_cornell_example(dataDir, dataType, split=None):
    # Open TFRecords file
    filename = '{}/{}_{}.tfrecords'.format(dataDir, dataType, split[:split.index('_')])
    check_filename(filename)
    writer = tf.python_io.TFRecordWriter(filename)

    path = "{}/{}/".format(dataDir, dataType)
    folders = os.listdir(path)
    if split != None:
        msplit = loadmat("{}/{}_split.mat".format(dataDir, dataType))
        folders = [sp[0] for sp in msplit[split][0]]

    for fid, folder in enumerate(folders):
        print('[{}/{}]'.format(str(fid+1), str(len(folders))), end='\r')
        sys.stdout.flush()
        if not os.path.isdir(path + folder + '/rgbjpg'):
            error('No "rgbjpg" folder in ' + path + folder)

        # Load annotations
        body_mat = loadmat(path + folder + '/body.mat')
        nb_frames, nb_body = body_mat['body'].shape
        height, width, channels = None, None, None
        # ft_list = []
        classes = load_classes(dataDir, dataType, folder)
        feature_images = []
        scene_joints = []
        scene_trackingstates = []
        scene_bodies = []
        for f in range(nb_frames):
            trackingstates = []
            joints_list = []
            bodies = []
            # Load image
            img_fpath = path + folder + '/rgbjpg/' + str(f + 1).zfill(4) + '.jpg'
            if not os.path.isfile(img_fpath):
                error('No such file: ' + img_fpath)
                continue
            imshape, image = load_image(img_fpath)
            if None in [height, width, channels]:
                height, width, channels = imshape

            # Load joints coordinates
            for b in range(nb_body):
                isBodyTracked = body_mat['body'][f, b][0, 0][0][0][0]
                if isBodyTracked != 1:
                    continue
                tst, joints = load_joints(f, b, body_mat)
                joints_list.append(joints)
                trackingstates.append(tst)
                bodies.append(b)
            # Features
            if len(bodies) < 1:
                bodies = [-1]
            joints_list = np.array(joints_list, dtype=np.float32).flatten()
            trackingstates = np.array(trackingstates, dtype=np.int64).flatten()
            bodies = np.array(bodies, dtype=np.int64).flatten()
            bytes_image = tf.compat.as_bytes(image)
            feature_image = bytes_feature(bytes_image)
            feature_images.append(feature_image)
            scene_joints.append(joints_list)
            scene_trackingstates.append(trackingstates)
            scene_bodies.append(bodies)
            # ft_dict = gen_feature2(feature_image, joints_list, trackingstates,
            #                        classes[f], bodies)
            # feature = tf.train.Features(feature=ft_dict)
            # ft_list.append(feature)
        # Sequence Example
        context = tf.train.Features(feature={
            'name': bytes_feature(folder.encode('utf-8')),
            'nb_frames': int64_feature(nb_frames),
            'height': int64_feature(height),
            'width': int64_feature(width)
        })
        # Feature lists
        fl_classes = tf.train.FeatureList(feature=[int64_feature(c) for c in classes])
        fl_tstates = tf.train.FeatureList(feature=[int64_feature_list(ts) for ts in scene_trackingstates])
        fl_bodies = tf.train.FeatureList(feature=[int64_feature_list(b) for b in scene_bodies])
        fl_images = tf.train.FeatureList(feature=feature_images)
        fl_joints = tf.train.FeatureList(feature=[float_feature_list(j) for j in scene_joints])
        feature_list = tf.train.FeatureLists(feature_list={
            'classes': fl_classes,
            'trackingStates': fl_tstates,
            'bodies': fl_bodies,
            'images': fl_images,
            'joints': fl_joints
        })
        ex = tf.train.SequenceExample(context=context, feature_lists=feature_list)
        # Serialize
        writer.write(ex.SerializeToString())
    # Close writer
    writer.close()
    print('\n')
    return


# Variables
# anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
dataDir = '/home/amusaal/DATA/Coco'
dataType = 'val2014'
parser()

if "Coco" in dataDir:
    gen_coco_example(dataDir, dataType)
elif "Cornell" in dataDir:
    gen_cornell_example(dataDir, dataType, 'train_name')
    gen_cornell_example(dataDir, dataType, 'test_name')

sys.stdout.flush()
