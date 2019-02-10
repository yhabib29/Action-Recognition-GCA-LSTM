import os
import sys
import scipy.io as sio
import tensorflow as tf
from ST_LSTM import stlstm_loop #STLSTMCell, STLSTMStateTuple
import numpy as np
import cv2


# ------------------------
#   HYPERPARAMETERS
# ------------------------


# WIDTH = 480
# HEIGHT = 480
CHANNELS = 3
BATCH_SIZE = 1


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
    images = tf.map_fn(tf.image.decode_jpeg, features['images'], dtype=tf.uint8)
    images = tf.map_fn(lambda x: tf.reverse(x, axis=[-1]), images, dtype=tf.uint8)
    # images = features['images']
    # joints = tf.sparse.to_dense(features['joints'])
    # trackingStates = tf.sparse.to_dense(features['trackingStates'])
    # bodies = tf.sparse.to_dense(features['bodies'], default_value=-1)
    bodies = tf.cast(features['bodies'], tf.int32)
    aclasses = tf.cast(features['classes'], tf.int32)
    framename = tf.cast(ctx['name'], tf.string)
    height = tf.cast(ctx['height'], tf.int32)
    width = tf.cast(ctx['width'], tf.int32)
    nb_frames = tf.cast(ctx['nb_frames'], tf.int32)

    joints = tf.sparse.to_dense(features['joints'])
    trackingStates = tf.sparse.to_dense(features['trackingStates'])
    joints = tf.reshape(joints, [nb_frames,25,3])
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
    return (images, aclasses, joints, trackingStates, bodies, height, width, nb_frames, framename)


def parse_labels(classes, scene_joints, scene_trackingStates, scene_bodies, nframe):
    labels = []
    for n in range(nframe):
        label = []
        nbody = len(scene_bodies[n]) if scene_bodies[n][0] == -1 else 0

        label.append(classes[n])
    return labels


def parse_body(bodies_data):
    """

    :param bodies_data: List of bodies index in each frame (-1 if nobody)
    :return:            List of bodies id in each frame reshaped
    """
    body_list = []
    for b in range(len(bodies_data)):
        if bodies_data[b] == -1:
            body_list.append([-1])
        bd = []
        nbb = 0
        for nb in range(6):
            if b + nbb >= len(bodies_data):
                continue
            if bodies_data[b + nbb] == nb:
                bd.append(nb)
                nbb += 1
        body_list.append(bd)
    return body_list


def parse_joints(joints_data, tstates_data, bodies_data):
    joints_list = []
    tstates_list = []
    j = 0
    while j < len(joints_data):
        frame_bodies = bodies_data[j]
        if frame_bodies == [-1]:
            joints_list.append([])
            j += 1
            continue
        nbody = len(frame_bodies)
        ts = []
        for b in range(nbody):
            jt = [joints_data[j][3*jo:3*jo+3] for jo in range(25)]
            ts.append(tstates_data[j])
            joints_list.append(jt)
            j += 1
        tstates_list.append(ts)
    return joints_list


def parse_data(joints_data, bodies_data):
    """

    :param joints_data:
    :param bodies_data:
    :return: joints_list    np.array of shape (body,frame,joint,3)
    """
    joints_dict = {}
    # joints_list = []
    # tstates_list = []
    b, fr = 0, 0
    while b < len(bodies_data):
        if bodies_data[b] == -1:
            fr += 1
            b += 1
            continue
        # bd = []
        nbb = 0
        for nb in range(6):
            if b + nbb >= len(bodies_data):
                nbb += 1
                continue
            if bodies_data[b + nbb] == nb:
                if not nb in joints_dict.keys():
                    joints_dict[nb] = fr * [[0,0,0]]
                if len(joints_dict[nb]) != fr:
                    joints_dict[nb] += (fr-len(joints_dict[nb])) * [[0,0,0]]
                # jt = [joints_data[b + nbb][3 * jo:3 * jo + 3].tolist() for jo in range(25)]
                jt = joints_data[fr][nbb*25:(nbb+1)*25].tolist()
                joints_dict[nb].append(jt)
                # bd.append(b)
                nbb += 1
        fr += 1
        b += nbb
        # b += 1
    nk = list(joints_dict.keys())
    joints_list = np.zeros((len(nk), fr, 25, 3))
    for e,k in enumerate(nk):
        # joints_list[e, :, :, :] = np.array(joints_dict[k])
        joints_list[e,:,:,:] = np.resize(np.array(joints_dict[k]), (fr,25,3))
    return joints_list

# ------------------------
#          NETWORK
# ------------------------


def build_lstm(lstm_sizes, inputs, keep_prob_, batch_size):
    """
    Create the LSTM layers
    """
    # lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
    # lstms = [STLSTMCell(size) for size in lstm_sizes]
    lstms = [tf.nn.rnn_cell.LSTMCell(size) for size in lstm_sizes]

    # Add dropout to the cell
    # drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

    # Stack up multiple LSTM layers, for deep learning
    # cell = tf.contrib.rnn.MultiRNNCell(lstms)
    cell = tf.nn.rnn_cell.MultiRNNCell(lstms)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
    # sc = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([1, lstm_sizes[0]], tf.float32),tf.zeros([1, lstm_sizes[0]], tf.float32))
    # sh = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([1, lstm_sizes[0]], tf.float32),tf.zeros([1, lstm_sizes[0]], tf.float32))
    # initial_state = (tf.zeros([1, lstm_sizes[0]*2], tf.float32),tf.zeros([1, lstm_sizes[0]*2], tf.float32))

    # lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    return initial_state, lstm_outputs, final_state, inputs


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


# Define variables
# inputs = tf.placeholder(tf.float32, (None, None, 3))  # (time, batch, features, channels) - (time,batch,in)
inputs = tf.placeholder(tf.float32, (BATCH_SIZE, None, 25, 3))  # (time, batch, features, channels) - (time,batch,in)
# outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)


# ------------------
# Define the graph
# ------------------
# init_state, outputs, final_state, inp = build_lstm([16], inputs, None, BATCH_SIZE)
outputs, states = stlstm_loop([16,16], inputs, do_norm=True) # do_norm=True
# Classif
# Loss

# Trainer - Backward propagation


# Add the variable initializer Op.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Create the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Summary
# writer = tf.summary.FileWriter('./log2', sess.graph)
with tf.variable_scope("ST-LSTM", reuse=tf.AUTO_REUSE):
    weights_summary = tf.summary.histogram('Weights', tf.get_variable("layer1/kernel_layer1"))
    weights_summary2 = tf.summary.histogram('Weights2', tf.get_variable("layer2/kernel_layer2"))


# Run the session
sess.run(init_op)
sess.run(tfrecord_iterator.initializer)


# TEST
# imgs, jts, aclasses, bds, tstates = sess.run(next_element)
# print('Images:\t\t',imgs.shape)
# print('Joints:\t\t',jts.shape)
# print('Classes:\t',aclasses.shape)
# print('Bodies:\t\t',bds.shape)
# print('TStates:\t',tstates.shape)

# Load scene
imgs, ac, jts, tStates, bds, h, w, nbf, fname = sess.run(next_element)
# Pre-process
fname = fname.decode('UTF-8')
print(jts.shape)
jts = parse_data(jts, bds)
# bds = parse_body(bds)
# jts = np.array(parse_joints(jts, tStates, bds))
print(imgs.shape)
print(h,w,nbf)
print(fname)
print(jts.shape)
# print(jts_dict[list(jts_dict.keys())[0]].shape)
# print('tStates',tStates.shape, tStates)
# print('Bodies', np.array(bds).shape, bds)
# print('Classes', ac.shape, ac)
# print('Joints', jts.shape, jts)

# For now use only one body
jts = jts[0]
if len(jts.shape) == 3:
    jts = jts.reshape((1,) + jts.shape)
    # jts = np.swapaxes(jts,0,1)
    # jts = np.swapaxes(jts, 1, 2)
print(jts.shape)    # shape = (frames,25,batch,3)
# init, out, fin, inps = sess.run([init_state, outputs, final_state, inp], feed_dict={inputs:jts[0]})
# print('\n\n',inps.shape,np.array(out).shape)
out, sta = sess.run([outputs, states], feed_dict={inputs:jts})
print(out.shape,sta.shape)


# Write summary
# (wsummary, wsummary2) = sess.run([weights_summary, weights_summary2])
# writer.add_summary(wsummary,1)
# writer.add_summary(wsummary2,1)

# cv2.imwrite('test/{}.jpg'.format(fname), imgs[0])
# video = cv2.VideoWriter('test/{}.avi'.format(fname), 0, 1, (w,h))
# for im in imgs:
#     video.write(im)
# video.release()



# Close the session
sess.close()
