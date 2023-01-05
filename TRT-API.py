from roboflow import Roboflow
import os, glob
import sys
import json  
import requests
import base64

local_inference_server_address = "http://localhost:9001/"
version_number = 10
project_id = "small-test-set/"
images_folder = "Images/"

# grab all the .jpg files
extention_images = ".jpg"
get_images = sorted(glob.glob(images_folder + '*' + extention_images))

# loop through all the images in the current folder
for images in get_images:

    # print file path
    print("File path: " + images)

    with open(images, "rb") as f:
        im_bytes = f.read()        
    im_b64 = base64.b64encode(im_bytes).decode("utf8")
    
    payload = json.dumps({"image": im_b64})

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    params = {
        'api_key': 'API',
    }

    response = requests.post(local_inference_server_address+project_id+str(version_number), params=params, headers=headers, data=im_b64)
    
    try:
        data = response.json()
    except:
        pass

    # print full json response
    print(data)

    for predictions in data['predictions']:
        
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