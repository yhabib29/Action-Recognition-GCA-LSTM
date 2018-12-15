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
DEPTH = 3
CHANNELS = 3
BATCH_SIZE = 4
ANCHORS = 5
GRID_WIDTH = 7
GRID_HEIGHT = 7
LEAKY = 0.1

# !TODO: One-hot-encoding
CLASSES = 80


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

    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                   target_height=HEIGHT,
                                                   target_width=WIDTH)
    # image = tf.reshape(image, [BATCH_SIZE, HEIGHT, WIDTH, DEPTH])
    #!TODO: Normaliser images = /255.0

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
def variables_yolo(init):
    variables = {}
    W = [None] * 23
    B = [None] * 23
    grid_depth =  ANCHORS * (CLASSES + 5)

    # Block 1
    W[0] = tf.get_variable("w1", [3, 3, 3, 32], initializer=init)
    B[0] = tf.get_variable("b1", [32], initializer=init)
    W[1] = tf.get_variable("w2", [3, 3, 32, 64], initializer=init)
    B[1] = tf.get_variable("b2", [64], initializer=init)
    W[2] = tf.get_variable("w3", [3, 3, 64, 128], initializer=init)
    B[2] = tf.get_variable("b3", [128], initializer=init)
    W[3] = tf.get_variable("w4", [1, 1, 128, 64], initializer=init)
    B[3] = tf.get_variable("b4", [64], initializer=init)
    W[4] = tf.get_variable("w5", [3, 3, 64, 128], initializer=init)
    B[4] = tf.get_variable("b5", [128], initializer=init)

    # Block 2
    W[5] = tf.get_variable("w6", [3, 3, 128, 256], initializer=init)
    B[5] = tf.get_variable("b6", [256], initializer=init)
    W[6] = tf.get_variable("w7", [1, 1, 256, 128], initializer=init)
    B[6] = tf.get_variable("b7", [128], initializer=init)
    W[7] = tf.get_variable("w8", [3, 3, 128, 256], initializer=init)
    B[7] = tf.get_variable("b8", [256], initializer=init)

    # Block 3
    W[8] = tf.get_variable("w9", [3, 3, 256, 512], initializer=init)
    B[8] = tf.get_variable("b9", [512], initializer=init)
    W[9] = tf.get_variable("w10", [1, 1, 512, 256], initializer=init)
    B[9] = tf.get_variable("b10", [256], initializer=init)
    W[10] = tf.get_variable("w11", [3, 3, 256, 512], initializer=init)
    B[10] = tf.get_variable("b11", [512], initializer=init)
    W[11] = tf.get_variable("w12", [1, 1, 512, 256], initializer=init)
    B[11] = tf.get_variable("b12", [256], initializer=init)
    W[12] = tf.get_variable("w13", [3, 3, 256, 512], initializer=init)
    B[12] = tf.get_variable("b13", [512], initializer=init)

    # Block 4
    W[13] = tf.get_variable("w14", [3, 3, 512, 1024], initializer=init)
    B[13] = tf.get_variable("b14", [1024], initializer=init)
    W[14] = tf.get_variable("w15", [1, 1, 1024, 512], initializer=init)
    B[14] = tf.get_variable("b15", [512], initializer=init)
    W[15] = tf.get_variable("w16", [3, 3, 512, 1024], initializer=init)
    B[15] = tf.get_variable("b16", [1024], initializer=init)
    W[16] = tf.get_variable("w17", [1, 1, 1024, 512], initializer=init)
    B[16] = tf.get_variable("b17", [512], initializer=init)
    W[17] = tf.get_variable("w18", [3, 3, 512, 1024], initializer=init)
    B[17] = tf.get_variable("b18", [1024], initializer=init)
    W[18] = tf.get_variable("w19", [3, 3, 1024, 1024], initializer=init)
    B[18] = tf.get_variable("b19", [1024], initializer=init)
    W[19] = tf.get_variable("w20", [3, 3, 1024, 1024], initializer=init)
    B[19] = tf.get_variable("b20", [1024], initializer=init)

    # Block 5
    W[20] = tf.get_variable("w21", [1, 1, 512, 64], initializer=init)
    B[20] = tf.get_variable("b21", [64], initializer=init)
    W[21] = tf.get_variable("w22", [3, 3, 1280, 1024], initializer=init)
    B[21] = tf.get_variable("b22", [1024], initializer=init)
    W[22] = tf.get_variable("w23", [1, 1, 1024, grid_depth], initializer=init)
    B[22] = tf.get_variable("b23", [grid_depth], initializer=init)

    for wk in range(1, len(W) + 1):
        wk_name = "w" + str(wk)
        bk_name = "b" + str(wk)
        variables[wk_name] = W[wk-1]
        variables[bk_name] = B[wk-1]
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
        bn = tf.contrib.layers.batch_norm(z)
        a = tf.nn.leaky_relu(bn, LEAKY)
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
    return tf.nn.max_pool(x, [1, size, size, 1], [1, stride, stride, 1], padding=pad, name=name)


def passthrough(x, p, kernel, bias, stride, size, name):
    cl = conv(p, kernel, bias, stride, name)
    cl = tf.space_to_depth(cl, size)
    y = tf.concat([x, cl], axis=3)
    return y


def loss():
    return


def yolo(data, vars):
    # Block 1
    x = conv(data, vars["w1"], vars["b1"], 1, "conv1")
    x = maxpool(x, 2, 2, "pool1")           # Pooling

    x = conv(x, vars["w2"], vars["b2"], 1, "conv2")
    x = maxpool(x, 2, 2, "pool2")

    x = conv(x, vars["w3"], vars["b3"], 1, "conv3")
    x = conv(x, vars["w4"], vars["b4"], 1, "conv4")
    x = conv(x, vars["w5"], vars["b5"], 1, "conv5")
    x = maxpool(x, 2, 2, "pool3")

    # Block 2
    x = conv(x, vars["w6"], vars["b6"], 1, "conv6")
    x = conv(x, vars["w7"], vars["b7"], 1, "conv7")
    x = conv(x, vars["w8"], vars["b8"], 1, "conv8")
    x = maxpool(x, 2, 2, "pool4")

    # Block 3
    x = conv(x, vars["w9"], vars["b9"], 1, "conv9")
    x = conv(x, vars["w10"], vars["b10"], 1, "conv10")
    x = conv(x, vars["w11"], vars["b11"], 1, "conv11")
    x = conv(x, vars["w12"], vars["b12"], 1, "conv12")
    pl = conv(x, vars["w13"], vars["b13"], 1, "conv13")
    x = maxpool(pl, 2, 2, "pool5")

    # Block 4
    x = conv(x, vars["w14"], vars["b14"], 1, "conv14")
    x = conv(x, vars["w15"], vars["b15"], 1, "conv15")
    x = conv(x, vars["w16"], vars["b16"], 1, "conv16")
    x = conv(x, vars["w17"], vars["b17"], 1, "conv17")
    x = conv(x, vars["w18"], vars["b18"], 1, "conv18")
    x = conv(x, vars["w19"], vars["b19"], 1, "conv19")
    x = conv(x, vars["w20"], vars["b20"], 1, "conv20")
    x = passthrough(x, pl, vars["w21"], vars["b21"], 1, 2, "conv21")
    x = conv(x, vars["w22"], vars["b22"], 1, "conv22")
    x = conv(x, vars["w23"], vars["b23"], 1, "conv23")

    dshape = (-1, GRID_HEIGHT, GRID_WIDTH, ANCHORS, CLASSES + 5)
    y = tf.reshape(x, shape=dshape, name="detection")

    return y


# ------------------------
#          TRAIN
# ------------------------

# TFRecords dataset paths
train_dataset = "../DATA/Coco/train2014.tfrecords"
valid_dataset = "../DATA/Coco/val2014.tfrecords"
# filename_queue = tf.train.string_input_producer([valid_dataset], num_epochs=1)
# image, label, bboxes, nb_objects = read_and_decode(filename_queue)

tfrecord_dataset = tf.data.TFRecordDataset(valid_dataset)
tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(True)
# tfrecord_dataset = tfrecord_dataset.repeat()
# tfrecord_dataset = tfrecord_dataset.batch(BATCH_SIZE)
pad_shapes = ([None, None, DEPTH], [1], [1], [1], [None], [None, 4])
tfrecord_dataset = tfrecord_dataset.padded_batch(BATCH_SIZE, padded_shapes=pad_shapes)
tfrecord_iterator = tfrecord_dataset.make_initializable_iterator()
next_element = tfrecord_iterator.get_next()

# Add the variable initializer Op.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# Weights
w_init = tf.contrib.layers.xavier_initializer()

# Create a saver for writing training checkpoints.
# weights_file = "yolo.weights"
# saver = tf.train.Saver(weights_file)


# Network (Training)
image = tf.placeholder(shape=[None, HEIGHT, WIDTH, DEPTH], dtype=tf.float32, name='image_placeholder')
# label = tf.placeholder(shape=[None, GRID_H, GRID_W, N_ANCHORS, 6], dtype=tf.float32, name='label_palceholder')
variables = variables_yolo(w_init)
predictions = yolo(image, variables)

# Create the session
sess = tf.Session()

# Run the session
sess.run(init_op)
sess.run(tfrecord_iterator.initializer)
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

imgs, H, W, NO, lbls, bbs = sess.run(next_element)

# _, loss_data, data = sess.run([train_step, loss, y], feed_dict={train_flag: True, image: image_data, label: label_data})
pred = sess.run([predictions], feed_dict={image: imgs})
print(pred)

"""for i in range(1):
    print('Current batch')
    img, lbl, bbox, nbo = imgs[5], lbls[5], bbs[5], NO[5]
    # img, lbl, bbox, nbo = sess.run([image, label, bboxes, nb_objects])
    for bb in bbox:
        x, y = bb[0], bb[1]
        w, h = bb[2], bb[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imwrite("test/img.jpg", img)
    plt.imsave("test/img.jpg", img)"""

writer = tf.summary.FileWriter('./log/yolo', sess.graph)

# Intialize iterator with training data
# sess.run(iterator.initializer, feed_dict={filenames: train_dataset})

coord.request_stop()
coord.join(threads)

# Close the session
sess.close()
