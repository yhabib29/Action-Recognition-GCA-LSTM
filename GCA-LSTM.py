import os
import sys
import scipy.io as sio
import tensorflow as tf
from ST_LSTM import stlstm_loop, stlstm_loss #STLSTMCell, STLSTMStateTuple
import numpy as np
from shutil import copy2
from datetime import datetime
import cv2


# ------------------------
#   HYPERPARAMETERS
# ------------------------


# WIDTH = 480
# HEIGHT = 480
CHANNELS = 3
BATCH_SIZE = 1
# NB_CLASSES = 10
LEARNING_RATE = 0.0015
ITERS = 10000   # 10000
NUM_UNITS = [128,128]
JOINTS = 16
GCA_KINECT = [1,20,3,8,9,10,4,5,6,0,16,17,18,12,13,14]


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


def convertJoints(joints):
    new_joints = [0] * JOINTS
    zero3D = [0.,0.,0.]
    for j in range(JOINTS):
        joint = joints[GCA_KINECT[j]]
        if joint != zero3D:
            new_joints[j] = joint
            continue
        else:
            if j == 2:
                joint = joints[2]
            elif j == 12:
                joint = joints[19]
            elif j == 15:
                joint = joints[15]
            elif j == 5:
                for jj in [11,24,23]:
                    joint = joints[jj]
                    if joint != zero3D:
                        break
            elif j == 8:
                for jj in [7,22,21]:
                    joint = joints[jj]
                    if joint != zero3D:
                        break
            new_joints[j] = joints
    return joints




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
                jt = convertJoints(jt)
                joints_dict[nb].append(jt)
                # bd.append(b)
                nbb += 1
        fr += 1
        b += nbb
        # b += 1
    nk = list(joints_dict.keys())
    joints_list = np.zeros((len(nk), fr, JOINTS, 3))
    for e,k in enumerate(nk):
        # joints_list[e, :, :, :] = np.array(joints_dict[k])
        joints_list[e,:,:,:] = np.resize(np.array(joints_dict[k]), (fr,JOINTS,3))
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





# TFRecords dataset paths
dataset_name = "office"
train_dataset = "../DATA/Cornell/{}_train_cornell.tfrecords".format(dataset_name)
valid_dataset = "../DATA/Cornell/{}_test_cornell.tfrecords".format(dataset_name)
# filename_queue = tf.train.string_input_producer([valid_dataset], num_epochs=1)
if "office" in dataset_name:
    NB_CLASSES = 10
    class_ids = [0,1,9,17,18,19,22,37,41,42]
    classnames = ['reading','walking','leave-office','fetch-book','put-back-book',
                  'put-down-item','take-item','play-computer','turn-on-monitor',
                  'turn-off-monitor']
elif "kitchen" in dataset_name:
    NB_CLASSES = 11
    class_ids = [0,1,2,4,6,7,8,9,10,13,15]
    classnames = ['fetch-from-fridge','put-back-to-fridge','prepare-food','microwaving',
                  'fetch-from-oven','pouring','drinking','leave-kitchen','fill-kettle',
                  'plug-in-kettle','move-kettle']
else:
    error("Use office or kitchen datasets only !")


tfrecord_dataset = tf.data.TFRecordDataset(train_dataset)
tfrecord_dataset = tfrecord_dataset.shuffle(buffer_size=1000)
tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(True)
tfrecord_dataset = tfrecord_dataset.repeat()
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
inputs = tf.placeholder(tf.float32, (BATCH_SIZE, None, JOINTS, 3))  # (time, batch, features, channels) - (time,batch,in)
loss = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="loss_placeholder")
pl_accuracy = tf.placeholder(shape=[], dtype=tf.float32, name="accuracy_placeholder")
ground_truth = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.int32, name="ground_truth_placeholder")
# outputs = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)


# ------------------
# Define the graph
# ------------------
# init_state, outputs, final_state, inp = build_lstm([16], inputs, None, BATCH_SIZE)
outputs = stlstm_loop(NUM_UNITS, inputs, NB_CLASSES, 2, do_norm=True) # do_norm=True
# Loss
# loss = stlstm_loss(outputs, ground_truth, NB_CLASSES)
loss_list = [stlstm_loss(out, ground_truth, NB_CLASSES) for out in reversed(outputs)]   # From last to first

# Trainer - Backward propagation
# opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)       # or use GradientDescentOptimizer
# update_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# with tf.control_dependencies(update_ops):
# train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
train_steps = [tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=lo, var_list=tf.trainable_variables())
               for lo in loss_list]


# Add the variable initializer Op.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Create the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Summary
varnames = ["ST-LSTM/layer1/kernel", "ST-LSTM/layer2/kernel", "ST-LSTM/GCACell/We1",
            "ST-LSTM/GCACell/We2","ST-LSTM/kernel_F1", "ST-LSTM/kernel_F2", "ST-LSTM/Wc"]
sm_loss = tf.summary.scalar(name='AVG Loss', tensor=tf.reduce_mean(loss_list))
sm_accuracy = tf.summary.scalar(name='Accuracy', tensor=pl_accuracy)
writer = tf.summary.FileWriter('./log2', sess.graph)
with tf.variable_scope("ST-LSTM", reuse=tf.AUTO_REUSE):
    # weights_summary = tf.summary.histogram('Weights', tf.get_variable("layer1/kernel"))
    # weights_summary2 = tf.summary.histogram('Weights2', tf.get_variable("layer2/kernel"))
    weights_summaries = [tf.summary.histogram(vname, tf.get_variable(vname[8:])) for vname in varnames]

# Save / Restore weights
log_file = 'log_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
weights_file = "gca_lstm"
saver = tf.train.Saver(max_to_keep=4)
# if os.path.isfile('./gca_lstm.ckpt.index'):
#     saver.restore(sess, tf.train.latest_checkpoint('./gca_lstm.ckpt'))

# Run the session
sess.run(init_op)
sess.run(tfrecord_iterator.initializer)

total_accuracy = 0.0
total_count = 0
for i in range(1,ITERS+1):
    log = ''

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
    ac = np.array(ac) - 1
    try:
        jts = parse_data(jts, bds)
    except ValueError:
        warning("Issue while parsing {}".format(fname))
        continue
    # bds = parse_body(bds)
    # jts = np.array(parse_joints(jts, tStates, bds))
    # print(imgs.shape)
    # print(h,w,nbf)
    # print(fname)
    # print(jts.shape)
    # print(jts_dict[list(jts_dict.keys())[0]].shape)
    # print('tStates',tStates.shape, tStates)
    # print('Bodies', np.array(bds).shape, bds)
    # print('Classes', ac.shape, ac)
    # print('Joints', jts.shape, jts)

    print("Iter {}: {} [{},{}] - {}".format(i,fname,w,h,jts.shape))

    # For now use only one body
    jts = jts[0]
    if len(jts.shape) == 3:
        jts = jts.reshape((1,) + jts.shape)
        # jts = np.swapaxes(jts,0,1)
        # jts = np.swapaxes(jts, 1, 2)
    # print(jts.shape)    # shape = (frames,25,batch,3)
    # init, out, fin, inps = sess.run([init_state, outputs, final_state, inp], feed_dict={inputs:jts[0]})
    # print('\n\n',inps.shape,np.array(out).shape)
    # out = sess.run(outputs, feed_dict={inputs:jts})

    start, end = 0, 1
    losses = []
    avg_loss = 0.0
    accuracy, count = 0.0, 0
    # Select all frames sequences label per label
    for k in range(1,len(ac)):
        end = k
        if k+1 == len(ac):
            end = k+1
        elif ac[k] == ac[k-1]:
            continue
        if ac[k] == -1:
            start = k
            continue
        indata = jts[:,start:end,:,:]
        gt = np.reshape(class_ids.index(int(ac[k])), (1,1))
        # out = sess.run(outputs, feed_dict={inputs: indata})
        # print(out)
        results, lo, _, sm_lo, sm_weights = sess.run([outputs, loss_list, train_steps, sm_loss, weights_summaries],
                               feed_dict={inputs: indata, ground_truth: gt})
        losses.append(lo[0])
        predicted_class = classnames[results[-1].argmax()]
        gtruth_class = classnames[class_ids.index(ac[k])]
        # Print predictions and save to log
        # print("Predicted = {} / Truth = {}  \tScores={}".format(class_ids[results[-1].argmax()], ac[k], results[-1]))
        # print("Predicted = {} / Truth = {}  \tScores={}".format(predicted_class,gtruth_class, results[-1]))
        line = "[{}] Predicted = {} / Truth = {}  \tScores={}".format(end-start, results[-1].argmax(),
                                                                     class_ids.index(ac[k]), results[-1].tolist())
        print(line)
        log += line + '\n'
        start = k
        # Accuracy
        count += 1
        if results[-1].argmax() == class_ids.index(ac[k]):
            accuracy += 1

    total_accuracy += accuracy
    total_count += count
    accuracy = accuracy / count
    avg_loss = np.array(losses).mean()
    # Print Loss and Accuracy and save to log
    line = "AVG Loss = {}\nAccuracy = {}\n".format(avg_loss, accuracy)
    print(line)
    with open(log_file, 'a') as flog:
        flog.write(log + line + '\n')
    if i%10 == 0:
        sm_acc = sess.run(sm_accuracy, feed_dict={pl_accuracy:accuracy})
        writer.add_summary(sm_acc, global_step=i)
        writer.add_summary(sm_lo, global_step=i)
        for sm_w in sm_weights:
            writer.add_summary(sm_w, global_step=i)
    if i%1000 == 0:
        saver.save(sess, "./{}".format(weights_file), global_step=i)



# Write summary
# (wsummary, wsummary2) = sess.run([weights_summary, weights_summary2])
# writer.add_summary(wsummary,1)
# writer.add_summary(wsummary2,1)

# cv2.imwrite('test/{}.jpg'.format(fname), imgs[0])
# video = cv2.VideoWriter('test/{}.avi'.format(fname), 0, 1, (w,h))
# for im in imgs:
#     video.write(im)
# video.release()

# Print Trainable variables
# variables_names = [v.name for v in tf.trainable_variables()]
# values = sess.run(variables_names)
# for k, v in zip(variables_names, values):
#     print("Variable: ", k)
#     print("Shape: ", v.shape)
#     print(v)

# Close the session
sess.close()

# Copy weights
nbt = 1
for dir in os.listdir('weights'):
    if not os.path.isdir('weights/{}'.format(dir)):
        continue
    if 'Train_' in dir:
        dir += 1
os.makedirs('weights/Train_{:03d}'.format(nbt))
for fi in os.listdir('.'):
    if os.path.isfile(fi) and weights_file in fi:
        copy2(fi, 'weights/Train_{:03d}/'.format(nbt))

print('Done')