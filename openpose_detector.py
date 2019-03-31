import sys
import cv2
import os
from time import time
ODIR = '/home/amusaal/Yassine/openpose'
sys.path.append(ODIR + '/build/python/openpose/')

try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')


# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.25
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 1
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = ODIR + "/models/"

# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

i=0
while i < 1:
    # Read new image
    # img = cv2.imread(ODIR + "/examples/media/COCO_val2014_000000000192.jpg")
    img = cv2.imread('../DATA/Cornell/kitchen/data_02-05-47/rgbjpg/0017.jpg')

    # Output keypoints and the image with the human skeleton blended on it
    t_start = time()
    keypoints, output_image = openpose.forward(img, True)
    t_end = time()

    # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    print(keypoints)

    # Display the image
    # cv2.imshow("output", output_image)
    # cv2.waitKey(15)

    # Save th eimage
    cv2.imwrite("test/openpose_output.jpg", output_image)
    i += 1

    # Time
    print('Time: {} s'.format(t_end - t_start))