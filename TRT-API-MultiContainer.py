from roboflow import Roboflow
import os, glob
import sys
import json  
import requests
import base64

# Establish the number of containers you are running
number_of_containers = 4

# Used for looping through containers - Don't change
counter_for_containers = 0

# Setting up arrays for Roboflow model varaibles
version_number_array = [10, 3, 3, 13]
project_id_array = ["small-test-set/", "car-models-nured/", "oxford-pets/", "hard-hat-workers/"]
images_folder_array = ["images\osprey", "images\cars", "images\pets", "images\safety-vests"]
api_key_array = ['API', 'API', 'API', 'API']

while counter_for_containers < number_of_containers:

    print("ACCESSING CONTAINER #"+str(counter_for_containers+1))

    local_inference_server_address = "http://localhost:900" + str(counter_for_containers+1) +"/"
    version_number = version_number_array[counter_for_containers]
    project_id = project_id_array[counter_for_containers]
    images_folder = images_folder_array[counter_for_containers]

    # grab all the .jpg files
    extention_images = ".jpg"
    get_images = sorted(glob.glob(images_folder + '/*' + extention_images))

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
            'api_key': str(api_key_array[counter_for_containers]),
        }

        response = requests.post(local_inference_server_address+project_id+str(version_number), params=params, headers=headers, data=im_b64)
        
        try:
            data = response.json()
        except Exception as e:
            print(e)
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
        
    # Used for incrementing port number and closing while loop
    counter_for_containers += 1