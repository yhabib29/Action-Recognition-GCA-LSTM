import os
import sys
import scipy.io as sio
import numpy as np
import cv2


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

def load_joints(f,b, bmat):
    jts = []
    for j in range(25):
        jt = {}
        jt['trackingState'] = bmat['body'][f,b][0, 0][1][0, j][0][0][0][0][0]
        jt['pcloud'] = bmat['body'][f,b][0, 0][1][0, j][0][0][5][0].tolist()
        jts.append(jt)
    return jts


DATADIR = "/home/amusaal/DATA/Cornell/"
dataset_name = 'office'
path = DATADIR + dataset_name + '/'

if False:
    folders = os.listdir(path)
    print("Loading Dataset !")
    for fid,folder in enumerate(folders):
        print('[' + str(fid) + '/' + str(len(folders)) + ']')
        sys.stdout.flush()
        if not os.path.isdir(path + folder + '/rgbjpg'):
            warning('No "rgbjpg" folder in ' + path + folder)
            continue
        # img_list = os.listdir(path + folder + '/rgbjpg/')
        body_mat = sio.loadmat(path + folder + '/body.mat')
        nb_frames, nb_body = body_mat['body'].shape
        for f in range(nb_frames):
            bodies = []
            # Load image
            img_fpath = path + folder + '/rgbjpg/' + str(f+1).zfill(4) + '.jpg'
            if not os.path.isfile(img_fpath):
                error('No such file: ' + img_fpath)
                continue
            img = cv2.imread(img_fpath)
            # Load joints coordinates
            for b in range(nb_body):
                isBodyTracked = body_mat['body'][f, b][0, 0][0][0][0]
                if isBodyTracked != 1:
                    continue
                joints = load_joints(f,b, body_mat)
                bodies.append(joints)








# BODY
body_mat = sio.loadmat(DATADIR + 'tools/data_sample/body.mat')
nb_frames, nb_body = body_mat['body'].shape

data = []
for f in range(nb_frames):
    bodies = []
    for b in range(nb_body):
        body = {}
        body['isBodyTracked'] = body_mat['body'][f,b][0,0][0][0][0]
        if body['isBodyTracked'] != 1:
            continue
        joints = []
        for j in range(25):
            joint = {}
            joint['trackingState'] = body_mat['body'][f,b][0,0][1][0,j][0][0][0][0][0]
            joint['camera'] = [cm[0] for cm in body_mat['body'][f,b][0,0][1][0,j][0][0][1]]
            joint['color'] = [cl[0] for cl in body_mat['body'][f,b][0,0][1][0,j][0][0][2]]
            joint['depth'] = [dp[0] for dp in body_mat['body'][f,b][0, 0][1][0, j][0][0][3]]
            joint['rotation'] = [rt[0] for rt in body_mat['body'][f,b][0, 0][1][0, j][0][0][4]]
            joint['pcloud'] = body_mat['body'][f,b][0, 0][1][0, j][0][0][5][0].tolist()
            joints.append(joint)
        # body['joints'] = joints
        bodies.append(body)
    data.append(bodies)

print(data)

# print(body_mat['body'][0,1][0,0][0][0][0])   # isBodyTracked
# print(body_mat['body'][0,1][0,0][1][0,1])     # joints (shape = (1,25))
# print(body_mat['body'][0,1][0,0][1][0,0][0,0][0][0][0])    # trackingState
# print(body_mat['body'][0,1][0,0][1][0,1][0][0][1])    # camera
# print(body_mat['body'][0,1][0,0][1][0,1][0][0][2])    # color
# print(body_mat['body'][0,1][0,0][1][0][0][0][0][3])    # depth
# print(body_mat['body'][0,1][0,0][1][0][0][0][0][4])    # rotation
# print(body_mat['body'][0,1][0,0][1][0][0][0][0][5][0])    # pcloud


# GND
gnd_mat = sio.loadmat(DATADIR + 'tools/data_sample/gnd.mat')
gnd = [gn[0] for gn in gnd_mat['gnd']]
print(gnd)    # gnd


depth_mat = sio.loadmat(DATADIR + 'tools/data_sample/depth/0001.mat')
dmap = np.array(depth_mat['depth'])
# cv2.imwrite("test/depth.jpg", dmap)
print(dmap.shape)     #depth map


img = cv2.imread(DATADIR + 'tools/data_sample/rgbjpg/0001.jpg')
# cv2.imwrite("test/action_img.jpg", img)
print(img.shape)     # img


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
