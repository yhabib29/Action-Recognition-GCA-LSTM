import os
import sys
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

# ------------------------
#   HYPERPARAMETERS
# ------------------------

WIDTH = 608
HEIGHT = 608
CHANNELS = 3
BATCH_SIZE = 32
ANCHORS = 5
GRID_WIDTH = 7
GRID_HEIGHT = 7
LEAKY = 0.1

# !TODO: One-hot-encoding
CLASS = 80


# ------------------------
#          DATA
# ------------------------


# Load data from Coco dataset
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
def load_data():
    global data_lbl, data_img
    # Collect data path (images and labels filepath)
    dataDir = '/home/amusaal/DATA/Coco'
    dataType = 'val2014'
    # Annotations
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)
    imgIds = coco.getImgIds()
    random.shuffle(imgIds)

    # print(coco.loadImgs([imgIds[0]]))
    # print(coco.loadAnns([annIds[0]]))

    # Collect images IDs and filenames
    print("Loading Images")
    for im in coco.loadImgs(imgIds):
        im_id = im['id']
        im_filename = im['file_name']
        data_img[im_id] = im_filename
        # Collect annotations
        annIds = coco.getAnnIds([im_id])
        data_lbl[im_id] = [[ann['category_id'], ann['bbox']] for ann in coco.loadAnns(annIds)]

    print("All data loaded")
    nb_img = len(data_img.keys())
    nb_ann = len(data_lbl.keys())
    return nb_img, nb_ann


def fill_feed_dict(images_pl, labels_pl, nb_images):
    """
    Set the feed_dict to fill the placeholder
    :param dataset:         Dataset
    :param images_pl:       Images placeholder
    :param labels_pl:       Labels placeholder
    :return:                feed_dict
    """
    global data_img, data_lbl
    img_keys = np.random.randint(0, nb_images, BATCH_SIZE)
    images_feed = []
    labels_feed = []
    for k in img_keys:
        img_file = tf.read_file(data_img[k])
        img_decoded = tf.image.decode_image(img_file, channels=3)
        lbls = data_lbl[k]
        images_feed.append(img_decoded)
        labels_feed.append(lbls)
    images_feed = tf.image.resize_images(images_feed, (WIDTH, HEIGHT))
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def read_and_decode(filename_queue):
    labels = []
    bboxes = []
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    """feature = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string)
        }"""
    feature = {'image': tf.FixedLenFeature((), tf.string),
               'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'objects_number': tf.FixedLenFeature([], tf.int64),
               'class': tf.VarLenFeature(tf.int64),
               'bboxes': tf.VarLenFeature(tf.float32)}
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert from a scalar string tensor
    # image = tf.decode_raw(features['image/encoded'], tf.uint8)
    # image = cv2.imdecode(features['image'], 3)
    # image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.image.decode_jpeg(features['image'], 3)
    bboxes = tf.sparse_tensor_to_dense(features['bboxes'], default_value=0)
    labels = tf.sparse_tensor_to_dense(features['class'], default_value=0)  # tf.decode_raw
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    nb_objects = tf.cast(features['objects_number'], tf.int32)
    is_object = tf.cast(nb_objects, tf.bool)

    image_shape = tf.stack([height, width, 3])
    bboxes_shape = tf.stack([nb_objects, 4])
    label_shape = tf.stack([nb_objects])  # ,1)

    image = tf.reshape(image, [height, width, 3])
    bboxes = tf.cond(is_object,
                     lambda: tf.reshape(bboxes, bboxes_shape),
                     lambda: tf.constant(-1.0))
    labels = tf.cond(is_object,
                     lambda: tf.reshape(labels, label_shape),
                     lambda: tf.constant(-1, dtype=tf.int64))

    """resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=HEIGHT,
                                                           target_width=WIDTH)

    images, labels = tf.train.shuffle_batch([resized_image, label],
                                                 batch_size=2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    """
    return image, labels, bboxes, nb_objects


def _parse_(serialized_example):
    feature = {'image': tf.FixedLenFeature((), tf.string),
               'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'objects_number': tf.FixedLenFeature([], tf.int64),
               'class': tf.VarLenFeature(tf.int64),
               'bboxes': tf.VarLenFeature(tf.float32)}
    features = tf.parse_single_example(serialized_example, feature)
    image = tf.image.decode_jpeg(features['image'], 3)
    bboxes = tf.sparse_tensor_to_dense(features['bboxes'], default_value=0)
    labels = tf.sparse_tensor_to_dense(features['class'], default_value=0)  # tf.decode_raw
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    nb_objects = tf.cast(features['objects_number'], tf.int32)
    is_object = tf.cast(nb_objects, tf.bool)

    image_shape = tf.stack([height, width, 3])
    bboxes_shape = tf.stack([nb_objects, 4])
    label_shape = tf.stack([nb_objects])  # ,1)

    image = tf.reshape(image, [height, width, 3])
    bboxes = tf.cond(is_object,
                     lambda: tf.reshape(bboxes, bboxes_shape),
                     lambda: tf.constant(-1.0))
    labels = tf.cond(is_object,
                     lambda: tf.reshape(labels, label_shape),
                     lambda: tf.constant(-1, dtype=tf.int64))

    # image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                                target_height=HEIGHT,
    #                                                target_width=WIDTH)

    return (image, [height], [width], [nb_objects], labels, bboxes)


def tfrecord_train_input_fn(tfrecord_path):
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_path)
    tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(True).batch(BATCH_SIZE)
    # tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
    tfrecord_iterator = tfrecord_dataset.make_initializable_iterator()

    return tfrecord_iterator.get_next()


# ------------------------
#       NETWORK
# ------------------------

# YOLO weights (filters + bias)
def variables_yolo():
    variables = {}
    W = []
    B = []

    # Block 1
    W[0] = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1))
    B[0] = tf.Variable(tf.zeros([32]))
    W[1] = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    B[1] = tf.Variable(tf.zeros([64]))
    W[2] = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
    B[2] = tf.Variable(tf.zeros([128]))
    W[3] = tf.Variable(tf.truncated_normal([1, 1, 128, 64], stddev=0.1))
    B[3] = tf.Variable(tf.zeros([64]))
    W[4] = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
    B[4] = tf.Variable(tf.zeros([128]))

    # Block 2
    W[5] = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
    B[5] = tf.Variable(tf.zeros([256]))
    W[6] = tf.Variable(tf.truncated_normal([1, 1, 256, 128], stddev=0.1))
    B[6] = tf.Variable(tf.zeros([128]))
    W[7] = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
    B[7] = tf.Variable(tf.zeros([256]))

    # Block 3
    W[8] = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
    B[8] = tf.Variable(tf.zeros([512]))
    W[9] = tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.1))
    B[9] = tf.Variable(tf.zeros([256]))
    W[10] = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
    B[10] = tf.Variable(tf.zeros([512]))
    W[11] = tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.1))
    B[11] = tf.Variable(tf.zeros([256]))
    W[12] = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
    B[12] = tf.Variable(tf.zeros([512]))

    # Block 4
    W[13] = tf.Variable(tf.truncated_normal([3, 3, 512, 1024], stddev=0.1))
    B[13] = tf.Variable(tf.zeros([1024]))
    W[14] = tf.Variable(tf.truncated_normal([1, 1, 1024, 512], stddev=0.1))
    B[14] = tf.Variable(tf.zeros([512]))
    W[15] = tf.Variable(tf.truncated_normal([3, 3, 512, 1024], stddev=0.1))
    B[15] = tf.Variable(tf.zeros([1024]))
    W[16] = tf.Variable(tf.truncated_normal([1, 1, 1024, 512], stddev=0.1))
    B[16] = tf.Variable(tf.zeros([512]))
    W[17] = tf.Variable(tf.truncated_normal([3, 3, 512, 1024], stddev=0.1))
    B[17] = tf.Variable(tf.zeros([1024]))
    W[18] = tf.Variable(tf.truncated_normal([3, 3, 1024, 1024], stddev=0.1))
    B[18] = tf.Variable(tf.zeros([1024]))
    W[19] = tf.Variable(tf.truncated_normal([3, 3, 1024, 1024], stddev=0.1))
    B[19] = tf.Variable(tf.zeros([1024]))

    # Block 5
    W[20] = tf.Variable(tf.truncated_normal([1, 1, 512, 64], stddev=0.1))
    B[20] = tf.Variable(tf.zeros([64]))
    W[21] = tf.Variable(tf.truncated_normal([3, 3, 1280, 1024], stddev=0.1))
    B[21] = tf.Variable(tf.zeros([1024]))
    W[22] = tf.Variable(tf.truncated_normal([1, 1, 1024, 425], stddev=0.1))
    B[22] = tf.Variable(tf.zeros([425]))

    for wk in range(len(W)):
        wk_name = "w" + str(wk)
        bk_name = "b" + str(wk)
        variables[wk_name] = W[wk]
        variables[bk_name] = B[wk]
    return variables


def conv(x, kernel, bias, stride, name, pad="SAME"):
    """
    Convolution Layer
    :param x:           Input data
    :param kernel:      Kernel (weights)
    :param bias:        Bias
    :param stride:      Stride
    :param name:        Name of the layer
    :param pad:         Padding
    :return:            Activation of the output of the convolution
    """
    with tf.name_scope(name):
        xW = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=pad)
        z = tf.nn.bias_add(xW, bias)
        a = tf.nn.leaky_relu(z, LEAKY)
    return (a)


def maxpool(x, size, stride, name, pad="SAME"):
    """
    Maxpooling Layer
    :param x:           Input data
    :param size:        Kernel size
    :param stride:      Stride
    :param name:        Name of the layer
    :param pad:         Padding
    :return:            Output of the maxpooling operation
    """
    return tf.nn.max_pool(x, size, stride, padding=pad, name=name)


def loss():
    return


def yolo(data, vars):

    # Conv1
    conv1 = conv(data, vars["w1"], vars["b1"], 1, "conv1")
    pool1 = maxpool(conv1, 2, 2, "pool1")           # Pooling
    bn1 = tf.contrib.layers.batch_norm(pool1)       # Batch Normalisation

    # Conv1
    conv1 = conv(data, vars["w1"], vars["b1"], 1, "conv1")
    pool1 = maxpool(conv1, 2, 2, "pool1")           # Pooling
    bn1 = tf.contrib.layers.batch_norm(pool1)       # Batch Normalisation

    return


# ------------------------
#          TRAIN
# ------------------------


"""
train_dataset = "../DATA/Coco/train2014.record"
valid_dataset = "../DATA/Coco/val2014.record"
with tf.Session() as sess:
    sess.run(init_op)
    features = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string)
        }
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([valid_dataset], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    data = tf.parse_single_example(serialized_example, features=features)

    # Convert the image data from string back to the numbers
    print("Decoding Image")
    image = tf.decode_raw(data['image/encoded'], tf.float32)
    print("Image decoded")
    cv2.imwrite("test/img.jpg", sess.run([image]))
    height = tf.cast(data['image/height'], tf.int32)
    width = tf.cast(data['image/width'], tf.int32)
    # Cast label data into int32
    label = tf.cast(data['image/object/class/label'], tf.int32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [WIDTH, HEIGHT, 3])

    # Creates batches by randomly shuffling tensors
    #images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
    #                                        min_after_dequeue=10)
"""

# Collect data path (images and labels filepath)
# data_lbl = {}
# data_img = {}
# nb_images, nb_labels = load_data()
#
# print("Images: ", nb_images)
# print("Labels: ", nb_labels)
# images = coco.loadImgs(imgIds[0:3])
# labels = coco.loadAnns(annIds[0:3])
# images = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds), size=batch_size)])
# print(len(images))
# print(labels[0])
# imtest_path = dataDir + '/' + dataType + '/' + images[0]['file_name']
# imtest = cv2.imread(imtest_path)
# cv2.imwrite("test/Test.jpg", imtest)
# print(imtest.shape)


# Initialize placeholders
# images = tf.placeholder(dtype = tf.int32, shape = [None, width, height, channels])
# labels = tf.placeholder(dtype = tf.int32, shape = [None, nb_classes])
# DATA_PATH = "/home/amusaal/DATA/Coco/"
# train_data_directory = os.path.join(DATA_PATH, "train2014")
# test_data_directory = os.path.join(DATA_PATH, "")
# image = tf.placeholder(shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32, name='images_placeholder')
# label = tf.placeholder(shape = [None, GRID_H, GRID_W, N_ANCHORS, 6], dtype=tf.float32, name='labels_palceholder')


# TFRecords dataset paths
train_dataset = "../DATA/Coco/train2014.tfrecords"
valid_dataset = "../DATA/Coco/val2014.tfrecords"
# filename_queue = tf.train.string_input_producer([valid_dataset], num_epochs=1)
# image, label, bboxes, nb_objects = read_and_decode(filename_queue)

tfrecord_dataset = tf.data.TFRecordDataset(valid_dataset)
tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(True)
# tfrecord_dataset = tfrecord_dataset.repeat()
# tfrecord_dataset = tfrecord_dataset.batch(BATCH_SIZE)
pad_shapes = ([None, None, 3], [1], [1], [1], [None], [None, 4])
tfrecord_dataset = tfrecord_dataset.padded_batch(BATCH_SIZE, padded_shapes=pad_shapes)
tfrecord_iterator = tfrecord_dataset.make_initializable_iterator()
next_element = tfrecord_iterator.get_next()

# Add the variable initializer Op.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# Weights initializer
w_init = tf.contrib.layers.xavier_initializer()

# Create a saver for writing training checkpoints.
# weights_file = "yolo.weights"
# saver = tf.train.Saver(weights_file)

# Create the session
sess = tf.Session()

# Run the session
sess.run(init_op)
sess.run(tfrecord_iterator.initializer)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

imgs, H, W, NO, lbls, bbs = sess.run(next_element)

for i in range(1):
    print('Current batch')
    img, lbl, bbox, nbo = imgs[5], lbls[5], bbs[5], NO[5]
    # img, lbl, bbox, nbo = sess.run([image, label, bboxes, nb_objects])
    for bb in bbox:
        x, y = bb[0], bb[1]
        w, h = bb[2], bb[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imwrite("test/img.jpg", img)
    plt.imsave("test/img.jpg", img)

# Intialize iterator with training data
# sess.run(iterator.initializer, feed_dict={filenames: train_dataset})

coord.request_stop()
coord.join(threads)

# Close the session
sess.close()
