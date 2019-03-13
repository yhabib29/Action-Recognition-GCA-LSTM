import os
import sys
import tensorflow as tf
from ST_LSTM import stlstm_loop, stlstm_loss #STLSTMCell, STLSTMStateTuple
import numpy as np
from shutil import copy2
from datetime import datetime
# import cv2
import traceback


# ------------------------
#   HYPERPARAMETERS
# ------------------------


CHANNELS = 3
# Batch size be 1 for now, because dataset sequences have different length.
# Or need to use padding and post-processing to use batch
BATCH_SIZE = 1
LEARNING_RATE = 0.0015
ITERS = 10000
NUM_UNITS = [128,128]
# Number of joints used by the network
JOINTS = 16
# Joints indexing correspondence table
GCA_KINECT = [1,20,3,8,9,10,4,5,6,0,16,17,18,12,13,14]


# ------------------------
#       TOOLS
# ------------------------


def isfloat(value):
    """
    Define if value can be convered to float
    :param value:   int or string to test
    :return:        True if it is a float, False if not
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def sigmoid(x, derivative=False):
    """
    Sigmoid function
    :param x:               Input value
    :param derivative:      Use derivative or not
    :return:                Sigmoid(x)
    """
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def warning(msg):
    """
    Tool to print Warning message
    :param msg:     Warning message to print
    """
    orange ='\033[33m'
    end = '\033[0m'
    print(orange + msg + end)
    return

def error(msg):
    """
    Tool to print error message and exit program
    :param msg:     Error message to print
    """
    red = '\033[31m'
    end = '\033[0m'
    print(red + msg + end)
    sys.exit(-1)
    return


def get_help():
    """
    Generate help message with program arguments
    :return:    Help message
    """
    gbold = '\033[1;32m'
    green = '\033[0;32m'
    dpath = '../DATA/Cornell/office_train.tfrecords'
    tpath = '../DATA/Cornell/office_test.tfrecords'
    cpath = '../DATA/Cornell/office.class'
    mhelp =  gbold + "--help [-h]\t\t\t" + green
    mhelp += "Show help\n"
    mhelp += gbold + "--mode [-m]\tMODE\t\t" + green
    mhelp += "Choose between Training ('train') and Test ('test') mode\n"
    mhelp += gbold + "--dataset [-d]\tPATH\t\t" + green
    mhelp += "Path to the TFRecords dataset or use 'office'/'kitchen' for Cornell dataset\n"
    mhelp += gbold + "--class [-c]\tPATH\t\t" + green
    mhelp += "Path to the class file containing class IDs and Class names where each line is: ID CLASS_NAME\n"
    mhelp += gbold + "--weights [-w]\tPATH\t\t" + green
    mhelp += "Output path without extension where to save/load the trained weights (default: './gca_lstm')\n"
    mhelp += gbold + "--dim [-n]\tDIM_LIST\t" + green
    mhelp += "LSTM layers dimension list (default: '128,128')\n"
    mhelp += gbold + "--lr [-l]\tLEARNING_RATE\t" + green
    mhelp += "Learning Rate\n"
    mhelp += gbold + "--iter [-i]\tITERATIONS\t" + green
    mhelp += "Number of training iterations\n\n"
    mhelp += gbold + "Example:\n" + green
    mhelp += "python3 GCA-LSTM.py -m train -l 0.0015 -i 10000 "
    mhelp += "-n 128,128 -d {} -c {} -w ./gca_lstm\n".format(dpath, cpath)
    mhelp += "python3 GCA-LSTM.py -m test -d {} -c {} -w ./gca_lstm\n\n".format(tpath, cpath)
    mhelp += '\033[0m'
    return mhelp


def read_dims(dims):
    """
    Parse LSTM dimensions argument
    :param dims:    String representing dimension list: 'dim1,dim2,dim3'
    :return:        List of LSTMs dimension
    """
    num_units_ = []
    lim = len(dims)
    start_ = 0
    for i,ch in enumerate(dims):
        if ch == ',' or i == lim-1:
            dimension = dims[start_: i] if i < lim-1 else dims[start_:]
            if dimension.isdigit():
                num_units_.append(int(dimension))
                start_ = i+1
                continue
        elif ch.isdigit():
            continue
        error('Error: Invalid character in {}\nIt should be for example: 128,128'.format(dims))
    return num_units_


def read_class(class_file_):
    """
    Parse class file where each line is: ID CLASS_NAME
    :param class_file:                      Path to the class file
    :return: nb_classes, cids, cnames       Number of classes, Class IDs, Class names
    """
    cids = []
    cnames = []
    nb_classes = 0
    with open(class_file_, 'r') as f:
        for l in f.readlines():
            lim = l.index(' ')
            cid = l[:lim]
            cname = l[lim + 1:-1]
            if not cid.isdigit():
                error('Invalid Class Index {} in {}'.format(cid, class_file_))
            if len(cname) < 1:
                error('Invalid class name {} in {}'.format(cname, class_file_))
            cids.append(int(cid))
            cnames.append(cname)
            nb_classes += 1
    return nb_classes, cids, cnames


# def get_weights(weights_file_):
#     """
#     Retrieve all weights having the prefix weights_file_
#     :param weights_file_:       Prefix of all weights files
#     :return: weights_files      List of weights
#     """
#     slash = weights_file_.rfind('/')
#     wdir = weights_file_[:slash+1] if slash != -1 else './'
#     wfile = weights_file_[slash+1:]
#     wlist = os.listdir(wdir)
#     for w in wlist:
#         if not wfile in w:
#             continue
#         if not w[-5:] == '.meta':
#             continue
#
#     return wfile_


def parse_args():
    """
    Parse program arguments
    """
    global ITERS, NB_CLASSES, NUM_UNITS, LEARNING_RATE
    global mode, dataset, dataset_name, class_ids, classnames, weights_file
    help_ = get_help()
    argv = sys.argv
    argc = len(argv)
    for a,arg in enumerate(argv):
        if a == argc-1:
            if arg in ['--help', '-h']:
                print(help_)
                sys.exit()
            break
        # Help
        elif arg in ['--help', '-h']:
            print(help_)
            sys.exit()
        # - Mode
        elif arg in ['--mode', '-m']:
            if argv[a+1] in ['train', 'test']:
                mode = argv[a+1]
            else:
                error('Error: Invalid mode {}'.format(argv[a+1]))
        # - Iterations
        elif arg in ['--iter', '-i']:
            if argv[a+1].isdigit():
                ITERS = int(argv[a+1])
            else:
                error('Error: Invalid number of iterations')
        # - Dataset
        elif arg in ['--dataset', '-d']:
            path = argv[a+1]
            if path in ['office', 'kitchen']:
                dataset_name = path
            elif os.path.isfile(path):
                dataset = path
            else:
                error('Error: Could not find dataset file at {}'.format(path))
        # - Class info
        elif arg in ['--class', '-c']:
            if os.path.isfile(argv[a+1]):
                [NB_CLASSES, class_ids, classnames] = read_class(argv[a+1])
            else:
                error('Error: Could not find class file {}'.format(argv[a+1]))
        # - Weights Path
        elif arg in ['--weights', '-w']:
            weights_file = argv[a+1]
        # - Learning Rate
        elif arg in ['--lr', '-l']:
            if isfloat(argv[a+1]):
                LEARNING_RATE = float(argv[a+1])
            else:
                error('Error: Invalid value for learning rate, it must be a float not {}'.format(argv[a+1]))
        # - LSTM dims
        elif arg in ['--dim', '-n']:
            NUM_UNITS = read_dims(argv[a+1])
        else:
            continue

    # Confirm Dataset path
    if not dataset:
        mode_ = 'test' if mode == 'valid' else mode
        dataset = "../DATA/Cornell/{}_{}.tfrecords".format(dataset_name, mode_)
        if None in [NB_CLASSES, class_ids, classnames]:
            class_file = "../DATA/Cornell/{}.class".format(dataset_name)
            [NB_CLASSES, class_ids, classnames] = read_class(class_file)
    if not os.path.isfile(dataset):
        warning('Could not find dataset at {}'.format(dataset))
        dataset = input('Enter dataset file path : ')
        if not os.path.isfile(dataset):
            error('Error: Could not find dataset at {}'.format(dataset))

    # Check weights file
    if mode == 'test':
        if not os.path.isfile('{}.meta'.format(weights_file)):
            error('Could not find the weights files at {}'.format(weights_file))

    return


def update_stats(stats_, label_, predictions_):
    """
    Update stats for test/validation mode.
    :param stats_:          Global stats array [TP,TN,FP,FN]
    :param label_:          Ground truth label
    :param predictions_:    List of predictions for each class
    :type stats_:           list
    :type label_:           int
    :type predictions_:     numpy.array
    :return:                Updated stats array
    """
    positive_class = predictions_.argmax()
    for n in range(len(stats_)):
        if n == positive_class:
            # True Positive
            if n == label_:
                stats_[n][0] += 1
            # False Positive
            else:
                stats_[n][2] += 1
        else:
            # True Negative
            if n != label_:
                stats_[n][1] += 1
            # False Negative
            else:
                stats_[n][3] += 1
    return stats_


def gen_order(gnd_classes):
    """
    Generate list of intervals to split the original sequence into variable length sequence with the same action.
    :param gnd_classes:     Ground Truth classes
    :return:                List of intervals
    """
    changes = [g for g in range(1,len(gnd_classes)) if gnd_classes[g-1] != gnd_classes[g]]
    order_ = []
    start = 0
    # Select all frames sequences label per label
    for g in changes:
        if gnd_classes[g-1] != -1:
            order_.append([start, g])
        start = g
    if gnd_classes[start] != -1:
        order_.append([start,len(gnd_classes)])
    # TODO: Remove this block
    # for k in range(1, len(gnd_classes)):
    #     # Move start to get only data with label
    #     if gnd_classes[k-1] == -1:
    #         start = k
    #         continue
    #     # Last frame
    #     if k + 1 == len(gnd_classes) and gnd_classes[k] != -1:
    #         end = k + 1
    #         order_.append([start, end])
    #     # If next frame is the same action
    #     elif gnd_classes[k] == gnd_classes[k - 1]:
    #         continue
    #     elif gnd_classes[k] != gnd_classes[k - 1]:
    #         end = k
    #         order_.append([start, end])
    #         start = k
    return order_



# ------------------------
#          DATA
# ------------------------


def _parse_(serialized_example):
    """
    Parse TFRecord dataset and extract data.
    :param serialized_example:  Serialized example of the TFRecord dataset (1 sequence)
    :return: (Images, Classes, Joints, trackingStates, Bodies, Height, Width, nb_frames, framename)
                Images:     Array of all sequence images
                Classes:    Array of all images activity class
                Joints:     3D coordinates of every bodies in each frame
                Bodies:     Array of bodies ID in each frame
                Height:     Height of sequence images
                Width:      Width of sequence images
                nb_frames:  Number of images in the sequence
                framename:  Name of the sequence
    """
    # Define context structure
    context = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'nb_frames': tf.FixedLenFeature([], dtype=tf.int64),
        'height': tf.FixedLenFeature([], dtype=tf.int64),
        'width': tf.FixedLenFeature([], dtype=tf.int64)
    }
    # Define feature structure
    feature = {'images': tf.FixedLenSequenceFeature([], dtype=tf.string),
               'classes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
               'bodies': tf.FixedLenSequenceFeature([], dtype=tf.int64),
               'joints': tf.VarLenFeature(dtype=tf.float32),
               'trackingStates': tf.VarLenFeature(dtype=tf.int64)
    }
    # Parse data from defined structures
    ctx,features = tf.parse_single_sequence_example(serialized_example,context,feature)
    # Images decoding
    images = tf.map_fn(tf.image.decode_jpeg, features['images'], dtype=tf.uint8)
    images = tf.map_fn(lambda x: tf.reverse(x, axis=[-1]), images, dtype=tf.uint8)
    # images = features['images']
    # Cast other data
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
    joints = tf.reshape(joints, [nb_frames, 25, 3])

    # return (image, joints, aclass, bodies, trackingStates)
    return (images, aclasses, joints, trackingStates, bodies, height, width, nb_frames, framename)


def convertJoints(joints_array):
    """
    TODO: Adapt this function to work with others indexing
    Convert Cornell joints indexing to GCA indexing.
    :param joints:  Joints coordinates array
    :return:        Converted joints array
    """
    jshape = (joints_array.shape[0], JOINTS, 3)
    new_joints_array = np.zeros((jshape))
    for f in range(len(joints_array)):
        joints = joints_array[f,:,:]
        new_joints = [0] * JOINTS
        for j in range(JOINTS):
            joint = joints[GCA_KINECT[j]]
            if np.any(joint != 0):
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
                        if np.any(joint != 0):
                            break
                elif j == 8:
                    for jj in [7,22,21]:
                        joint = joints[jj]
                        if np.any(joint != 0):
                            break
                new_joints[j] = joint
        new_joints_array[f,:,:] = new_joints
    return new_joints_array


def parse_data(joints_data, bodies_data):
    """
    TODO: Remove this function because the array structure has been fixed in conversation script
    Parse raw data loaded from TFRecord and extract structured joints lists.
    Each frame may contain multiple bodies. We have a list containing all the joints
    coordinates of the sequence. For each frame, we also have a list of bodies ID present in the image. If there is
    nobody, the list is [-1].
    :param joints_data:
    :param bodies_data:
    :return: joints_list    np.array of shape (body,frame,joint,3)
    """
    joints_dict = {}
    # joints_list = []
    # tstates_list = []
    b, fr = 0, 0
    while b < len(bodies_data):
        # Nobody in the image
        if bodies_data[b] == -1:
            fr += 1
            b += 1
            continue
        # bd = []
        # Initialize the frame bodies number
        nbb = 0
        # TODO: Double check if there are 6 or 7 bodies
        # TODO: Check if there is one class per body per frame
        for nb in range(6):
            # Test if we will exceed the sequence bodies list
            if b + nbb >= len(bodies_data):
                nbb += 1
                continue
            if bodies_data[b + nbb] == nb:
                # If it is the first time we see the body, we add zero coords for previous frames
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
        try:
            joints_list[e,:,:,:] = np.resize(np.array(joints_dict[k]), (fr,JOINTS,3))
        except ValueError:
            print(joints_dict[k])
            sys.exit(-11)
    return joints_list


# ------------------------
#          TRAIN
# ------------------------


def train(log_file_):
    global NUM_UNITS, NB_CLASSES, BATCH_SIZE, JOINTS, LEARNING_RATE, ITERS, weights_file
    # Define variables
    # inputs = tf.placeholder(tf.float32, (None, None, 3))  # (time, batch, features, channels) - (time,batch,in)
    inputs = tf.placeholder(tf.float32, (BATCH_SIZE, None, JOINTS, 3))  # (time, batch, features, channels)
    loss = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.float32, name="loss_placeholder")
    pl_accuracy = tf.placeholder(shape=[], dtype=tf.float32, name="accuracy_placeholder")
    ground_truth = tf.placeholder(shape=[BATCH_SIZE, 1], dtype=tf.int32, name="ground_truth_placeholder")

    # Create the graph of the network
    outputs = stlstm_loop(NUM_UNITS, inputs, NB_CLASSES, 2, do_norm=True)  # do_norm=True

    # Loss
    # TODO: Train loss on the whole sequence
    # loss = stlstm_loss(outputs, ground_truth, NB_CLASSES)
    # loss_list = [stlstm_loss(out, ground_truth, NB_CLASSES) for out in reversed(outputs)]  # From last to first
    loss_list = [stlstm_loss(out, ground_truth, NB_CLASSES) for out in outputs]  # From first to last

    # Trainer - Backward propagation
    # train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    train_steps = [tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=lo, var_list=tf.trainable_variables())
                   for lo in loss_list]

    # Add the variable initializer Op.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Summary - Tensorboard variables
    writer = tf.summary.FileWriter('./log2', sess.graph)
    sm_loss = tf.summary.scalar(name='AVG Loss', tensor=tf.reduce_mean(loss_list))
    # TODO: automatic generation of varnames
    # varnames = ["ST-LSTM/layer1/kernel", "ST-LSTM/layer2/kernel", "ST-LSTM/GCACell/We1",
    #             "ST-LSTM/GCACell/We2", "ST-LSTM/kernel_F1", "ST-LSTM/kernel_F2", "ST-LSTM/Wc"]
    sm_accuracy = tf.summary.scalar(name='Accuracy', tensor=pl_accuracy)
    # with tf.variable_scope("ST-LSTM", reuse=tf.AUTO_REUSE):
    #     weights_summaries = [tf.summary.histogram(vname, tf.get_variable(vname[8:])) for vname in varnames]
    merged_summary_op = tf.summary.merge_all()

    # Saver to save weights
    saver = tf.train.Saver(max_to_keep=10)

    # restore weights if needed Fine-Tuning
    # if os.path.isfile('./gca_lstm.ckpt.index') and mode in ['test', 'valid']:
    #     saver.restore(sess, tf.train.latest_checkpoint('./gca_lstm.ckpt'))

    # Run the session
    sess.run(init_op)
    sess.run(tfrecord_iterator.initializer)

    total_accuracy = 0.0
    total_count = 0
    for i in range(1, ITERS + 1):
        log = ''

        # DEBUG - Data loading
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

        if bds.shape[0] != jts.shape[0]:
            warning('Bodies and joints data have different length (not handled yet)')
            continue

        # Convert joints
        jts = convertJoints(jts)

        # TODO: Remove this block
        # try:
        #     jts = parse_data(jts, bds)
        #     continue
        # except ValueError:
        #     warning("Issue while parsing {}".format(fname))
        #     traceback.print_exc()
        #     sys.exit(-10)

        # DEBUG - Parsing data
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

        print("Iter {}: {} [{},{}] - {}".format(i, fname, w, h, jts.shape))
        log += "Iter {}: {} [{},{}] - {}\n".format(i, fname, w, h, jts.shape)

        # For now use only one body
        if len(jts.shape) == 3:
            jts = jts.reshape((1,) + jts.shape)
            # jts = np.swapaxes(jts,0,1)
            # jts = np.swapaxes(jts, 1, 2)
        # print(jts.shape)    # shape = (frames,25,batch,3)

        # Loop vars
        start, end = 0, 1  # size of window
        losses = []
        avg_loss = 0.0
        accuracy, count = 0.0, 0
        # Generate order (variable sequence length)
        order = gen_order(ac)
        # Select all frames sequences label per label
        for start, end in order:

            # Window - subset of data
            indata = jts[:, start:end, :, :]
            print(start,end,indata.shape)
            # Ground Truth label
            gnd = ac[end-1]
            gt = np.reshape(class_ids.index(int(gnd)), (1, 1))
            # out = sess.run(outputs, feed_dict={inputs: indata})
            # print(out)

            # Train
            try:
                results, lo, _, sm_lo = sess.run([outputs, loss_list, train_steps, sm_loss],
                                             feed_dict={inputs: indata, ground_truth: gt})
            except tf.python.framework.errors_impl.InvalidArgumentError:
                print(gnd, gt, end, ac[end - 1:end + 1])
                sys.exit(-1)
            # results, lo, _, sm_lo, sm_weights = sess.run([outputs, loss_list, train_steps,
            #                                               sm_loss, weights_summaries],
            #                                              feed_dict={inputs: indata, ground_truth: gt})

            # Get loss and predictions
            losses.append(lo[-1])
            predicted_class = classnames[results[-1].argmax()]
            gtruth_class = classnames[class_ids.index(gnd)]
            # Print predictions and save to log
            # print("Predicted = {} / Truth = {}  \tScores={}".format(class_ids[results[-1].argmax()], gnd,
            #                                                         results[-1]))
            # print("Predicted = {} / Truth = {}  \tScores={}".format(predicted_class, gtruth_class,
            #                                                         results[-1]))
            line = "[{}] Predicted = {} / Truth = {}  \tScores={}".format(end - start,
                                                                          results[-1].argmax(),
                                                                          class_ids.index(gnd),
                                                                          results[-1].tolist())
            print(line)
            log += line + '\n'
            # start = k
            # Accuracy
            count += 1
            if results[-1].argmax() == class_ids.index(gnd):
                accuracy += 1

        # Compute average accuracy and average loss
        total_accuracy += accuracy
        total_count += count
        accuracy = accuracy / count
        avg_loss = np.array(losses).mean()

        # Print Loss and Accuracy and save to log
        line = "AVG Loss = {}\nAccuracy = {}\n".format(avg_loss, accuracy)
        print(line)
        with open(log_file_, 'a') as flog:
            flog.write(log + line + '\n')

        # Save weights, accuracy and loss values for tensorboard
        if i % 10 == 0:
            # sm_acc = sess.run(sm_accuracy, feed_dict={pl_accuracy:accuracy})
            # writer.add_summary(sm_acc, global_step=i)
            # writer.add_summary(sm_lo, global_step=i)
            # for sm_w in sm_weights:
            #     writer.add_summary(sm_w, global_step=i)
            merged_summary = sess.run(merged_summary_op, feed_dict={pl_accuracy: accuracy})
            writer.add_summary(merged_summary, global_step=i)

        # Save weights each 1000 iterations
        if i % 1000 == 0:
            saver.save(sess, "./{}".format(weights_file), global_step=i)

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
    copy2('checkpoint', 'weights/Train_{:03d}/'.format(nbt))
    print('Saved weights into weights/Train_{:03d}/'.format(nbt))

    return True


# ------------------------
#          TEST
# ------------------------


def test(log_file_):
    global NUM_UNITS, NB_CLASSES, BATCH_SIZE, JOINTS, weights_file
    # Define variables
    inputs = tf.placeholder(tf.float32, (BATCH_SIZE, None, JOINTS, 3))  # (time, batch, features, channels)
    pl_accuracy = tf.placeholder(shape=[], dtype=tf.float32, name="accuracy_placeholder")

    # Define the graph
    # TODO: Add args to change ST-LSTM hyperparameters
    outputs = stlstm_loop(NUM_UNITS, inputs, NB_CLASSES, 2, do_norm=True)  # do_norm=True

    # Create the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Summary - Tensorboard variables
    sm_accuracy = tf.summary.scalar(name='Total Accuracy', tensor=pl_accuracy)
    writer = tf.summary.FileWriter('./log_gca_test', sess.graph)
    merged_summary_op = tf.summary.merge_all()

    # Saver to restore weights
    saver = tf.train.Saver()
    if os.path.isfile('{}.index'.format(weights_file)):
        # saver.restore(sess, tf.train.latest_checkpoint('{}'.format(weights_file)))
        saver.restore(sess, '{}'.format(weights_file))
    else:
        error('Error: Could not find weights file {}'.format(weights_file))

    # Run the session
    sess.run(tfrecord_iterator.initializer)

    # Validation/Test variables
    stats = np.zeros((NB_CLASSES, 4))  # [TP,TN,FP,FN]

    total_accuracy = 0.0
    total_count = 0
    it = 1
    try:
        while True:
            log = ''

            # Load scene
            imgs, ac, jts, tStates, bds, h, w, nbf, fname = sess.run(next_element)

            # Pre-process
            fname = fname.decode('UTF-8')
            ac = np.array(ac) - 1

            if bds.shape[0] != jts.shape[0]:
                warning('Bodies and joints data have different length (not handled yet)')
                continue

            # Convert joints
            jts = convertJoints(jts)

            # TODO: Remove this block
            # try:
            #     jts = parse_data(jts, bds)
            # except ValueError:
            #     warning("Issue while parsing {}".format(fname))
            #     continue

            line = "Iter {}: {} [{},{}] - {}".format(it, fname, w, h, jts.shape)
            log += line + "\n"
            print(line)

            # For now use only one body
            if len(jts.shape) == 3:
                jts = jts.reshape((1,) + jts.shape)
                # jts = np.swapaxes(jts,0,1)
                # jts = np.swapaxes(jts, 1, 2)
            # print(jts.shape)    # shape = (frames,25,batch,3)

            # Loop vars
            # start, end = 0, 1  # size of window
            accuracy, count = 0.0, 0
            # Generate order (variable sequence length)
            order = gen_order(ac)
            # Select all frames sequences label per label
            for start,end in order:

                # Window - subset of data
                indata = jts[:, start:end, :, :]
                # Ground Truth label
                gnd = ac[end-1]
                gt = np.reshape(class_ids.index(int(gnd)), (1, 1))

                # Inference
                results = sess.run(outputs, feed_dict={inputs: indata})
                # print(out)

                # Show predictions
                # predicted_class = classnames[results[-1].argmax()]
                # gtruth_class = classnames[class_ids.index(gnd)]
                # Print predictions and save to log
                line = "[{}] Predicted = {} / Truth = {}  \tScores={}".format(end - start, results[-1].argmax(),
                                                                              class_ids.index(gnd),
                                                                              results[-1].tolist())
                print("[{}] Predicted = {} / Truth = {}".format(end - start,
                                                                results[-1].argmax(),
                                                                class_ids.index(gnd)))
                log += line + '\n'
                # start = k
                # print(line)

                # Accuracy
                count += 1
                if results[-1].argmax() == class_ids.index(gnd):
                    accuracy += 1

                # Update stats
                stats = update_stats(stats, class_ids.index(gnd), results[-1])

                # TODO: Make function to show predictions
                # cv2.imwrite('test/{}.jpg'.format(fname), imgs[0])
                # video = cv2.VideoWriter('test/{}.avi'.format(fname), 0, 1, (w,h))
                # for im in imgs:
                #     video.write(im)
                # video.release()

            # Compute average accuracy and average loss
            total_accuracy += accuracy
            total_count += count
            accuracy = accuracy / count

            # Print Loss and Accuracy and save to log
            line = "Sequence Accuracy = {:0.4f}".format(accuracy)
            print(line)
            with open(log_file_, 'a') as flog:
                flog.write(log + line + '\n')

            # Save accuracy for tensorboard
            if it % 1 == 0:
                # sm_acc = sess.run(sm_accuracy, feed_dict={pl_accuracy:accuracy})
                merged_summary = sess.run(merged_summary_op,
                                          feed_dict={pl_accuracy: total_accuracy / total_count})
                writer.add_summary(merged_summary, global_step=it)

            if it % 10 == 0:
                precision_ = stats[:, 0] / (stats[:, 0] + stats[:, 2])
                recall_ = stats[:, 0] / (stats[:, 0] + stats[:, 3])
                print('Recall = {}'.format(recall_.tolist()))
                print('Precision = {}\n'.format(precision_.tolist()))

            it += 1

    # End of the dataset
    except tf.errors.OutOfRangeError:
        pass

    # Print Recall and Accuracy
    total_accuracy = total_accuracy / total_count
    precision = stats[:,0] / (stats[:,0] + stats[:,2])
    recall = stats[:,0] / (stats[:,0] + stats[:,3])
    print('Recall = {}'.format(recall.tolist()))
    print('Precision = {}'.format(precision.tolist()))
    print('Total Accuracy: {}'.format(total_accuracy))

    # Close the session
    sess.close()

    return True


# ------------------------
#          SCRIPT
# ------------------------


# Parse args
mode = 'train'
dataset = None
dataset_name = "office"
weights_file = './gca_lstm'
NB_CLASSES = None
class_ids = None
classnames = None
parse_args()

# Print parameters
blue = '\33[0;36m'
bbold = '\33[1;36m'
print("{}PARAMETERS\n{}".format(bbold, blue))
print("{}MODE:{}\t\t{}".format(bbold, blue, mode))
print("{}DATASET:{}\t{}".format(bbold, blue, dataset))
print("{}WEIGHTS:{}\t{}".format(bbold, blue, weights_file))
print("{}CLASSES:{}\t{} classes - {}\n{}".format(bbold, blue, NB_CLASSES, class_ids, classnames))
print("{}LSTM DIMS:{}\t{}".format(bbold, blue, NUM_UNITS))
if mode == 'train':
    print("{}ITERS:{}\t\t{}".format(bbold, blue, ITERS))
    print("{}LEARNING RATE:{}\t{}".format(bbold, blue, LEARNING_RATE))
print('\033[0m\n\n\n')
if not input('Do you want to continue ? [y/n] ') in ['y', 'Y']:
    sys.exit()

# Load Data
# TFRecords dataset paths
# filename_queue = tf.train.string_input_producer([valid_dataset], num_epochs=1)


# Load data
tfrecord_dataset = tf.data.TFRecordDataset(dataset)
tfrecord_dataset = tfrecord_dataset.shuffle(buffer_size=1000)
tfrecord_dataset = tfrecord_dataset.map(lambda x: _parse_(x)).shuffle(True)
# Infinite loop in dataset
if mode == 'train':
    tfrecord_dataset = tfrecord_dataset.repeat()
# Read all the element only once
else:
    tfrecord_dataset = tfrecord_dataset.repeat(1)
# Padded batch
# tfrecord_dataset = tfrecord_dataset.batch(BATCH_SIZE)
# pad_shapes = (tf.TensorShape([None, None, CHANNELS]),
#               tf.TensorShape([None, 25, 3]),
#               tf.TensorShape([1]),
#               tf.TensorShape([]),
#               tf.TensorShape([None, 25]))
# pad_shapes = ([None, None, CHANNELS], [None, 25, 3], [1], [None], [None, 25])
# tfrecord_dataset = tfrecord_dataset.padded_batch(BATCH_SIZE, padded_shapes=pad_shapes)
# Iterator for reading data
tfrecord_iterator = tfrecord_dataset.make_initializable_iterator()
next_element = tfrecord_iterator.get_next()

# Create the log file
log_file = 'logs/log_{}.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

result = False
if mode == 'train':
    result = train(log_file)
elif mode == 'test':
    result = test(log_file)
else:
    sys.exit(-4)


print('Done !')