import os
import sys
import cv2
from time import time
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

# ------------------------
#   HYPERPARAMETERS
# ------------------------

WIDTH = 448
HEIGHT = 448
assert WIDTH % 32 == 0, 'Network input size should be mutliple of 32.'
assert HEIGHT % 32 == 0, 'Network input size should be mutliple of 32.'
CHANNELS = 3
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_ITERS = 45000
NB_ANCHORS = 5
GRID_WIDTH = WIDTH // 32
GRID_HEIGHT = HEIGHT // 32
LEAKY = 0.1
LCOORD = 5
LNOOBJ = 0.5
ANCHORS = np.array(
    [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])

# !TODO: One-hot-encoding
CLASSES = 80
# Coco class IDs
CLASSES_ID = [c for c in range(1, 91) if c not in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]]


# ------------------------
#       TOOLS
# ------------------------


def info(msg, end=None, std='\33[0m'):
    blue = '\33[36;1m'
    if end == None:
        print(blue + msg + std)
    else:
        print(blue + msg + std, end=end)
    return


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def yolo_ground_truth(batch_true_boxes, num_objects):
    """
    Function adapted from https://github.com/allanzelener/YAD2K/blob/master/yad2k/models/keras_yolo.py
    Convert ground truth annotation into yolo format
    :param true_boxes:  Original bounding boxes (np.array)
    :return: detection_map, gt_yolo_map
    """
    box_dim = batch_true_boxes.shape[2]
    detection_mask = np.zeros((BATCH_SIZE, GRID_HEIGHT, GRID_WIDTH, NB_ANCHORS, 1),
                              dtype=np.float32)
    gt_yolo_map = np.zeros((BATCH_SIZE, GRID_HEIGHT, GRID_WIDTH, NB_ANCHORS, box_dim),
                           dtype=np.float32)
    for ba, true_boxes in enumerate(batch_true_boxes):
        num_object = int(num_objects[ba])
        for nb in range(num_object):
            box = true_boxes[nb]
            # print("BOX", box)
            category_id, box = int(box[0]), box[1:]
            class_id = CLASSES_ID.index(category_id)
            best_iou = 0
            best_anchor = 0
            # box = box[0:4] * np.array([GRID_WIDTH / WIDTH, GRID_HEIGHT / HEIGHT,
            #                            GRID_WIDTH / WIDTH, GRID_HEIGHT / HEIGHT])
            box = box[0:4] * np.array([GRID_WIDTH, GRID_HEIGHT, GRID_WIDTH, GRID_HEIGHT])
            cx = int(box[0]) if (box[0] < GRID_WIDTH) else (GRID_WIDTH - 1)
            cy = int(box[1]) if (box[1] < GRID_HEIGHT) else (GRID_HEIGHT - 1)
            for k, anchor in enumerate(ANCHORS):
                box_maxes = box[2:4] / 2.0
                box_mins = - box_maxes
                abox_maxes = (anchor / 2.0)
                abox_mins = - abox_maxes

                intersect_mins = np.maximum(box_mins, abox_mins)
                intersect_maxes = np.minimum(box_maxes, abox_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]
                box_area = box[2] * box[3]
                anchor_area = anchor[0] * anchor[1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = k

            if best_iou > 0:
                detection_mask[ba, cy, cx, best_anchor] = 1
                # Removed np.log for box size (exp is used in loss calculation)
                yolo_box = np.array(
                    [box[0] - cx, box[1] - cy,
                     box[2] / ANCHORS[best_anchor][0],
                     box[3] / ANCHORS[best_anchor][1], class_id],
                    dtype=np.float32)
                # print("YOLO_BOX", yolo_box)
                gt_yolo_map[ba, cy, cx, best_anchor] = yolo_box
    return detection_mask, gt_yolo_map


def detect_box(dmap, im=None):
    #!TODO Check
    ims = []
    # print(dmap.shape)
    batch_bbs = []
    batch_len,gw,gh = dmap.shape[:3]
    for k in range(batch_len):
        imb = im[k].copy()
        bbs = []
        for cx in range(gw):
            for cy in range(gh):
                for a in range(5):
                    anchor = dmap[k,cy,cx,a]
                    if sigmoid(anchor[4]) > 0.5:
                        factor_w = WIDTH / GRID_WIDTH
                        factor_h = HEIGHT / GRID_HEIGHT
                        bx = (cx + sigmoid(anchor[0])) * factor_w
                        by = (cy + sigmoid(anchor[1])) * factor_h
                        bw = (ANCHORS[a][0] * np.exp(anchor[2])) * factor_w
                        bh = (ANCHORS[a][1] * np.exp(anchor[3])) * factor_h
                        cid = anchor[5:].argmax()
                        score = softmax(anchor[cid])
                        if not score > 0.5:
                            continue
                        lbl = CLASSES_ID[cid]
                        bbs.append([bx, by, bw, bh, score, lbl])
                        if im is not None:
                            imb = cv2.rectangle(imb,(int(bx),int(by)), (int(bw),int(bh)),2)
        ims.append(imb)
        batch_bbs.append(bbs)
    # print(np.array(bbs).shape)
    return batch_bbs, ims

# ------------------------
#          DATA
# ------------------------


def _parse_(serialized_example):
    # feature = {'image': tf.FixedLenFeature((), tf.string),
    #            'height': tf.FixedLenFeature([], tf.int64),
    #            'width': tf.FixedLenFeature([], tf.int64),
    #            'objects_number': tf.FixedLenFeature([], tf.int64),
    #            'bboxes': tf.VarLenFeature(tf.float32)}
    context = {'image': tf.FixedLenFeature((), tf.string),
               'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'objects_number': tf.FixedLenFeature([], tf.int64)}
    feature = {'bboxes': tf.VarLenFeature(dtype=tf.float32)}
    # features = tf.parse_single_example(serialized_example, feature)
    features, fbboxes = tf.parse_single_sequence_example(serialized_example, context, feature)
    image = tf.image.decode_jpeg(features['image'], 3)
    bboxes = tf.sparse_tensor_to_dense(fbboxes['bboxes'], default_value=0)
    # bboxes = tf.cast(fbboxes['bboxes'], tf.float32)
    # labels = tf.sparse_tensor_to_dense(features['class'], default_value=0)  # tf.decode_raw
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    nb_objects = tf.cast(features['objects_number'], tf.int32)
    is_object = tf.cast(nb_objects, tf.bool)

    # image_shape = tf.stack([height, width, 3])
    # bboxes_shape = tf.stack([nb_objects, 5])
    # label_shape = tf.stack([nb_objects])  # ,1)

    image = tf.reshape(image, [height, width, 3])
    # bboxes = tf.cond(is_object,
    #                  lambda: tf.reshape(bboxes, bboxes_shape),
    #                  lambda: tf.constant(-1.0))
    # labels = tf.cond(is_object,
    #                  lambda: tf.reshape(labels, label_shape),
    #                  lambda: tf.constant(-1, dtype=tf.int64))

    # d_map, gt_map = yolo_ground_truth(bboxes)
    image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                   target_height=HEIGHT,
                                                   target_width=WIDTH)
    # image = tf.reshape(image, [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])
    # !TODO: Normaliser images = /255.0

    # return (image, [height], [width], [nb_objects], bboxes)
    return (image, bboxes, [nb_objects])


def load_weights(wfile):
    # !TODO Load weights
    global variables, sess
    print("Loading Model ... ", end="")
    var_names = tf.contrib.framework.list_variables(wfile)
    # Collect batch normalization variables
    bn_vars = {}
    for va in tf.global_variables():
        varname = va.op.name
        if not 'batch_normalization' in varname:
            continue
        if not ('moving_mean' in varname or 'moving_variance' in varname):
            continue
        bid = varname[20:varname.index('/')]
        if bid == "":
            bid = "0"
        bid = int(bid) + 1
        bname = "bn" + str(bid)
        if not bname in bn_vars.keys():
            bn_vars[bname] = [None, None]
        if 'moving_mean' in varname:
            bn_vars[bname][0] = va
        elif 'moving_variance' in varname:
            bn_vars[bname][1] = va
    # Assign pre-trained weights values
    for name, shape in var_names:
        if 'weights' in name:
            wname = "w" + name[19:name.index('/')]
            var = tf.contrib.framework.load_variable(wfile, name)
            sess.run(variables[wname].assign(var))
        if 'moving_mean' in name:
            mname = "bn" + name[19:name.index('/')]
            var = tf.contrib.framework.load_variable(wfile, name)
            sess.run(bn_vars[mname][0].assign(var))
        if 'moving_variance' in name:
            vname = "bn" + name[19:name.index('/')]
            var = tf.contrib.framework.load_variable(wfile, name)
            sess.run(bn_vars[vname][1].assign(var))
    print("Done\n")
    return

# ------------------------
#       NETWORK
# ------------------------

# YOLO weights (filters + bias)
def variables_yolo(init):
    variables = {}
    W = [None] * 23
    B = [None] * 23
    grid_depth = NB_ANCHORS * (CLASSES + 5)

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
        variables[wk_name] = W[wk - 1]
        variables[bk_name] = B[wk - 1]
    return variables


def conv(x, kernel, bias, stride, name, batchnorm=True, pad="SAME"):
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
        z = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=pad)
        # z = tf.nn.bias_add(z, bias)
        # bn = tf.contrib.layers.batch_norm(z) if batchnorm else z
        bn = tf.layers.batch_normalization(z, training=True) if batchnorm else z
        a = tf.nn.leaky_relu(bn, LEAKY)
    return a


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


def yolo_loss(pred, detection_map, ground_truth):
    mask = detection_map
    label = ground_truth
    # mask = ground_truth[...,5:]
    # label = ground_truth[...,0:5]

    mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_HEIGHT, GRID_WIDTH, NB_ANCHORS)), tf.bool)

    with tf.name_scope('mask'):
        masked_label = tf.boolean_mask(label, mask)
        masked_pred = tf.boolean_mask(pred, mask)
        masked_pred_noobj = tf.boolean_mask(pred, tf.logical_not(mask))

    with tf.name_scope('pred'):
        masked_pred_xy = tf.sigmoid(masked_pred[..., 0:2])  # - cx/xy
        # !TODO Not exponent or remove log from GT
        masked_pred_wh = tf.sqrt(tf.exp(masked_pred[..., 2:4]))
        # masked_pred_wh = tf.sqrt(masked_pred[..., 2:4])
        masked_pred_o = tf.sigmoid(masked_pred[..., 4:])
        masked_pred_no_o = tf.sigmoid(masked_pred_noobj[..., 4:])
        masked_pred_c = tf.nn.softmax(masked_pred[..., 5:])

    with tf.name_scope('lab'):
        masked_label_xy = masked_label[..., 0:2]
        # !TODO: Log of masked_label
        masked_label_wh = tf.sqrt(masked_label[..., 2:4])
        masked_label_c = masked_label[..., 4:]
        masked_label_c_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_c, tf.int32), depth=CLASSES),
                                        shape=(-1, CLASSES))

    with tf.name_scope('merge'):
        with tf.name_scope('loss_xy'):
            loss_xy = tf.reduce_sum(tf.square(masked_pred_xy - masked_label_xy))
        with tf.name_scope('loss_wh'):
            loss_wh = tf.reduce_sum(tf.square(masked_pred_wh - masked_label_wh))
        with tf.name_scope('loss_obj'):
            loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1))
        with tf.name_scope('loss_no_obj'):
            loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o))
        with tf.name_scope('loss_class'):
            loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))

        loss = LCOORD * (loss_xy + loss_wh) + loss_obj + LNOOBJ * loss_no_obj + loss_c

    return loss, (loss_xy, loss_wh, loss_obj, loss_no_obj, loss_c)


def yolo(data, vars):
    # Block 1
    x = conv(data, vars["w1"], vars["b1"], 1, "conv1")
    x = maxpool(x, 2, 2, "pool1")  # Pooling

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
    x = conv(x, vars["w23"], vars["b23"], 1, "conv23", False)

    dshape = (-1, GRID_HEIGHT, GRID_WIDTH, NB_ANCHORS, CLASSES + 5)
    y = tf.reshape(x, shape=dshape, name="detection")

    return y


# ------------------------
#          TRAIN
# ------------------------

def describe_model():
    """ print a description of the current model parameters """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    msg = [""]
    total = 0
    for v in train_vars:
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        msg.append("{}: shape={}, dim={}".format(
            v.name, shape.as_list(), ele))
    size_mb = total * 4 / 1024.0**2
    msg.append("Total param={} ({:01f} MB assuming all float32)".format(total, size_mb))
    print("Model Parameters: {}".format('\n'.join(msg)))

def get_variables_values(sess):
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    values = {}
    for variable in variables:
        values[variable.name[:-2]] = sess.run(variable)
    return values

# TFRecords dataset paths
train_dataset = "../DATA/Coco/train2014_yolo.tfrecords"
valid_dataset = "../DATA/Coco/val2014_coco.tfrecords"
# filename_queue = tf.train.string_input_producer([valid_dataset], num_epochs=1)
# image, label, bboxes, nb_objects = read_and_decode(filename_queue)

info("Loading Dataset (TFRecords) ...")
# with tf.device('/cpu:0'):
tfrecord_dataset = tf.data.TFRecordDataset(valid_dataset)
tfrecord_dataset = tfrecord_dataset.shuffle(buffer_size=100*BATCH_SIZE)
tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x), num_parallel_calls=16)
# tfrecord_dataset = tfrecord_dataset.cache()
# tfrecord_dataset = tfrecord_dataset.repeat()
# tfrecord_dataset = tfrecord_dataset.batch(BATCH_SIZE)
# pad_shapes = ([None, None, CHANNELS], [1], [1], [1], [None], [None, 4])
pad_shapes = ([None, None, CHANNELS], [None, 5], [1])
# pad_shapes = ([None, None, CHANNELS], [GRID_HEIGHT, GRID_WIDTH, 5, 1], [GRID_HEIGHT, GRID_WIDTH, 5, 5])
tfrecord_dataset = tfrecord_dataset.padded_batch(BATCH_SIZE, padded_shapes=pad_shapes)
# tfrecord_dataset = tfrecord_dataset.prefetch(1)
tfrecord_iterator = tfrecord_dataset.make_initializable_iterator()
next_element = tfrecord_iterator.get_next()

# Add the variable initializer Op.
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Weights initializer
w_init = tf.contrib.layers.xavier_initializer()

# Network (Training)
dct_maps_pl = tf.placeholder(shape=[BATCH_SIZE, GRID_HEIGHT, GRID_WIDTH, NB_ANCHORS, 1],
                             dtype=np.int32, name="dmap_placeholder")
gt_yolo_maps_pl = tf.placeholder(shape=[BATCH_SIZE, GRID_HEIGHT, GRID_WIDTH, NB_ANCHORS, 5],
                                 dtype=np.float32, name="gt_placeholder")
image = tf.placeholder(shape=[BATCH_SIZE, HEIGHT, WIDTH, CHANNELS], dtype=tf.float32, name='image_placeholder')
loss = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="loss_placeholder")
# label = tf.placeholder(shape=[None, GRID_H, GRID_W, NB_ANCHORS, 6], dtype=tf.float32, name='label_palceholder')

# with tf.device('/gpu:1'):
info("Defining the graph ...")
variables = variables_yolo(w_init)
predictions = yolo(image, variables)
yolo_losses, etc = yolo_loss(predictions, dct_maps_pl, gt_yolo_maps_pl)
opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)       # or use GradientDescentOptimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = opt.minimize(yolo_losses)

# Create the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Create a saver for writing training checkpoints.
d1 = "../darkflow/built_graph/"
d2 = "../darknet/"
weights_file = "{}yolov2.ckpt".format(d2)
# weights_vars = {key:value for key,value in variables.items() if key[0] == "w"}
saver = tf.train.Saver(max_to_keep=4)
# saver.restore(sess, tf.train.latest_checkpoint(weights_file))


# Log
summary = tf.summary.scalar(name='Loss', tensor=yolo_losses)
summary_image = tf.summary.image("Images", image)


# Run the session
info('Initializing variables ... ', end='')
# sess.run(init_op)
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(tfrecord_iterator.initializer)

writer = tf.summary.FileWriter('./log/yolo', sess.graph)


# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Load weights
# load_weights(weights_file)

# DEBUG
# describe_model()
# print(get_variables_values(sess).keys())
# print("\n\n")
# key = "BatchNorm/moving_mean"
# with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
#     v = tf.get_variable(key, shape=(32,))
# print(v)
# print(sess.run(tf.get_variable('batch_normalization_15/moving_mean')))

# NUM_ITERS = 100

# input('Ready ?')
info("\n\nStarting training:")
for it in range(NUM_ITERS):
    # Load data
    # imgs, H, W, NO, lbls, bbs = sess.run(next_element)
    imgs, bbs, NO = sess.run(next_element)
    # imgs, detection_maps, ground_truth_yolo_maps = sess.run(next_element)
    # print(bbs.shape)

    # Ground truth maps
    # with tf.device('/cpu:0'):
    detection_maps, ground_truth_yolo_maps = yolo_ground_truth(np.array(bbs), NO)
    # print(np.array(detection_maps).shape)
    # print(np.array(ground_truth_yolo_maps).shape)

    # Forward
    # pred = sess.run(predictions, feed_dict={image: imgs})
    # db,di = detect_box(pred, imgs)
    # im_summary = sess.run(summary_image, feed_dict={image: di})
    # writer.add_summary(im_summary, global_step=it + 1)
    # print(np.array(pred).shape)

    # Loss
    t_start = time()
    _, summary_str, loss, letc = sess.run([train_step, summary, yolo_losses, etc],
                    feed_dict={image: imgs, dct_maps_pl: detection_maps, gt_yolo_maps_pl: ground_truth_yolo_maps})
    t_chrono = time() - t_start
    print('Time: {} s'.format(t_chrono))
    print("Iter {}:\t\tAvg Loss={}".format(it,loss / BATCH_SIZE))
    print("Losses ", letc)

    # if it + 1 % 10 == 0:
    writer.add_summary(summary_str, global_step=it + 1)
    if it + 1 % 1000 == 0:
        saver.save(sess, "./coco_yolov2", global_step=it + 1)

#!TODO Make detection from predictions

# for i in range(1):
#     # print('Current batch')
#     img, bbox = imgs[0], detect_box(pred)
#     # img, lbl, bbox, nbo = sess.run([image, label, bboxes, nb_objects])
#     for bb in bbox:
#         x, y = int(bb[0]), int(bb[1])
#         w, h = int(bb[2]), int(bb[3])
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     # cv2.imwrite("test/img.jpg", img)
#     plt.imsave("test/img.jpg", img)

# coord.request_stop()
# coord.join(threads)

# Save weights
save_path = saver.save(sess, "./coco_yolov2.ckpt")

# Close the session
sess.close()
