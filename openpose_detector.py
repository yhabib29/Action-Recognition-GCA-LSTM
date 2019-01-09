import sys
import cv2
import os
ODIR = '/home/amusaal/Yassine/openpose'
sys.path.append(ODIR + '/python')

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

while 1:
    # Read new image
    img = cv2.imread("../../../examples/media/COCO_val2014_000000000192.jpg")
    # Output keypoints and the image with the human skeleton blended on it
    keypoints, output_image = openpose.forward(img, True)
    # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    print(keypoints)
    # Display the image
    # cv2.imshow("output", output_image)
    # cv2.waitKey(15)
    # Save th eimage
    cv2.imwrite("output.jpg", output_image)