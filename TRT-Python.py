from roboflow import Roboflow
import os, glob

local_inference_server_address = "http://localhost:9001/"
version_number = 10

rf = Roboflow(api_key="API")
project = rf.workspace().project("small-test-set")
local_model = project.version(version_number=version_number, local=local_inference_server_address).model

images_folder = "Images/"

# grab all the .jpg files
extention_images = ".jpg"
get_images = sorted(glob.glob(images_folder + '*' + extention_images))

# loop through all the images in the current folder
for images in get_images:

    print("\n"+images+"\n")

    # infer on a local image
    inference = local_model.predict(images, confidence=40, overlap=30).json()

    # print full inference
    print(inference)

    for predictions in inference['predictions']:
        
        # set class name based on prediction
        Pred_name = predictions['class']
        # print(Pred_name)

        # set confidence based on prediction
        Pred_confidence = predictions['confidence']
        # print(Pred_confidence)

        # set bounding box height based on prediction
        Pred_height = predictions['height']
        # print(Pred_height)

        # set bounding box width based on prediction
        Pred_width = predictions['width']
        # print(Pred_width)

        # set bounding box x based on prediction
        Pred_x = predictions['x']
        # print(Pred_x)

        # set bounding box y based on prediction
        Pred_y = predictions['y']
        # print(Pred_y)

