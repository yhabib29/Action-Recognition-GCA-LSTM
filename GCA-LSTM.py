import os
import sys
import scipy.io as sio
import tensorflow as tf
import numpy as np
import cv2


# ------------------------
#   HYPERPARAMETERS
# ------------------------


# WIDTH = 480
# HEIGHT = 480
CHANNELS = 3
BATCH_SIZE = 200


# ------------------------
#       TOOLS
# ------------------------


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def warning(msg):
    orange ='\033[33m'
    end = '\033[0m'
    print(orange + msg + end)
    return

def error(msg):
    red = '\033[31m'
    end = '\033[0m'
    print(red + msg + end)
    sys.exit(-1)
    return


# ------------------------
#          DATA
# ------------------------


def _parse_(serialized_example):
    context = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'nb_frames': tf.FixedLenFeature([], dtype=tf.int64),
        'height': tf.FixedLenFeature([], dtype=tf.int64),
        'width': tf.FixedLenFeature([], dtype=tf.int64)
    }
    feature = {'images': tf.FixedLenSequenceFeature([], dtype=tf.string),
               'classes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
               'bodies': tf.FixedLenSequenceFeature([], dtype=tf.int64),
               'joints': tf.VarLenFeature(dtype=tf.float32),
               'trackingStates': tf.VarLenFeature(dtype=tf.int64)
               }
    ctx,features = tf.parse_single_sequence_example(serialized_example,context,feature)
    # images = [tf.image.decode_jpeg(im, 3) for im in features['images']]
    images = features['images']
    joints = tf.sparse_tensor_to_dense(features['joints'], default_value=0)
    # trackingStates = tf.sparse_tensor_to_dense(features['trackingStates'], default_value=0)
    # bodies = tf.sparse_tensor_to_dense(features['bodies'], default_value=0)
    # bodies = tf.cast(features['bodies'], tf.int32)
    # aclass = tf.cast(features['class'], tf.int32)
    framename = tf.cast(ctx['name'], tf.string)
    height = tf.cast(ctx['height'], tf.int32)
    width = tf.cast(ctx['width'], tf.int32)
    nb_frames = tf.cast(ctx['nb_frames'], tf.int32)
    # is_object = tf.cast(nb_bodies, tf.bool)

    # image_shape = tf.stack([height, width, CHANNELS])
    # nb_bodies = tf.shape(bodies)
    # print(nb_bodies)
    # joints_shape = tf.stack([nb_bodies, 25, 3])
    # label_shape = tf.stack([nb_objects])  # ,1)

    # image = tf.reshape(image, [height, width, 3])
    # joints = tf.cond(tf.reshape(joints, joints_shape))
    # labels = tf.cond(is_object,
    #                  lambda: tf.reshape(labels, label_shape),
    #                  lambda: tf.constant(-1, dtype=tf.int64))

    # d_map, gt_map = yolo_ground_truth(bboxes)
    # image = tf.image.resize_image_with_crop_or_pad(image=image,
    #                                                target_height=HEIGHT,
    #                                                target_width=WIDTH)
    # image = tf.reshape(image, [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS])

    # return (image, joints, aclass, bodies, trackingStates)
    return (images, height, width, nb_frames, framename)


def load_joints(f,b, bmat):
    jts = []
    for j in range(25):
        jt = {}
        jt['trackingState'] = bmat['body'][f,b][0, 0][1][0, j][0][0][0][0][0]
        jt['pcloud'] = bmat['body'][f,b][0, 0][1][0, j][0][0][5][0].tolist()
        jts.append(jt)
    return jts


"""
DATADIR = "/home/amusaal/DATA/Cornell/"
dataset_name = 'office'
path = DATADIR + dataset_name + '/'

# BODY
body_mat = sio.loadmat(DATADIR + 'tools/data_sample/body.mat')
nb_frames, nb_body = body_mat['body'].shape

oclasses_mat = sio.loadmat(DATADIR + 'office_classname.mat')
office_classes = []
for oc in oclasses_mat['office_classname'][0]:
    if oc.shape[0] != 1:
        office_classes.append(None)
    elif len(oc[0]) < 1:
        office_classes.append(None)
    else:
        office_classes.append(oc[0])
print(office_classes)
# print(len(oclasses_mat['office_classname'][0][2][0]))

kclasses_mat = sio.loadmat(DATADIR + 'kitchen_classname.mat')
kitchen_classes = []
for kc in kclasses_mat['kitchen_classname'][0]:
    if kc.shape[0] != 1:
        kitchen_classes.append(None)
    elif len(kc[0]) < 1:
        kitchen_classes.append(None)
    else:
        kitchen_classes.append(kc[0])
print(kitchen_classes)
# print(kclasses_mat['kitchen_classname'])
"""


# TFRecords dataset paths
train_dataset = "../DATA/Cornell/office_train_cornell.tfrecords"
valid_dataset = "../DATA/Cornell/office_test_cornell.tfrecords"
# filename_queue = tf.train.string_input_producer([valid_dataset], num_epochs=1)
# image, label, bboxes, nb_objects = read_and_decode(filename_queue)

tfrecord_dataset = tf.data.TFRecordDataset(train_dataset)
tfrecord_dataset = tfrecord_dataset.shuffle(buffer_size=10000)
tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(True)
# tfrecord_dataset = tfrecord_dataset.repeat()
# tfrecord_dataset = tfrecord_dataset.batch(BATCH_SIZE)
# pad_shapes = (tf.TensorShape([None, None, CHANNELS]),
#               tf.TensorShape([None, 25, 3]),
#               tf.TensorShape([1]),
#               tf.TensorShape([]),
#               tf.TensorShape([None, 25]))
# pad_shapes = ([None, None, CHANNELS], [None, 25, 3], [1], [None], [None, 25])
# tfrecord_dataset = tfrecord_dataset.padded_batch(BATCH_SIZE, padded_shapes=pad_shapes)
tfrecord_iterator = tfrecord_dataset.make_initializable_iterator()
next_element = tfrecord_iterator.get_next()

# Add the variable initializer Op.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



# Create the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Run the session
sess.run(init_op)
sess.run(tfrecord_iterator.initializer)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# TEST
# imgs, jts, aclasses, bds, tstates = sess.run(next_element)
# print('Images:\t\t',imgs.shape)
# print('Joints:\t\t',jts.shape)
# print('Classes:\t',aclasses.shape)
# print('Bodies:\t\t',bds.shape)
# print('TStates:\t',tstates.shape)
imgs, h, w, nbf, fname = sess.run(next_element)
print(imgs[0])
print(h,w,nbf)
print(fname)



coord.request_stop()
coord.join(threads)

# Close the session
sess.close()
