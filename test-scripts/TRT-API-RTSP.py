from roboflow import Roboflow
import os, glob
import cv2
import sys
import json  
import requests
import base64
import time

# local_inference_server_address = "http://detect.roboflow.com/"
local_inference_server_address = "http://localhost:9001/"
version_number = 14
project_id = "oak-test-objects/"

fps_array = []
counter = 0

vcap = cv2.VideoCapture("rtsp://192.168.1.17:554/s2")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontScale = 1
thickness = 1
color = (0, 200, 0)

box_color = (0, 200, 0)
box_thickness = 1 
box_scale = 1

while(1):
    
    t0 = time.time()

    counter += 1

    if counter % 4 == 0:
        vcap.grab()

    ret, frame = vcap.read()

    success, encoded_image = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(encoded_image).decode('utf-8')

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    params = {
        'api_key': 'API',
    }

    # print("PAYLOAD SENT")
    response = requests.post(local_inference_server_address+project_id+str(version_number), params=params, headers=headers, data=base64_image)
    # print(response)

    try:
        data = response.json()
    except Exception as e:
        print(e)
        pass

    # print full json response
    try:
        print(data)
    except Exception as e:
        print(e)
        pass

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

        x0 = predictions['x'] - predictions['width'] / 2
        y0 = predictions['y'] - predictions['height'] / 2
        x1 = predictions['x'] + predictions['width'] / 2
        y1 = predictions['y'] + predictions['height'] / 2
        box = (x0, y0, x1, y1)

        box_start_point = (int(x0), int(y0))
        box_end_point = (int(x1), int(y1))
        text_point = (int(x0), int(y0-20))

        image_drawn = cv2.rectangle(frame, box_start_point, box_end_point, box_color, box_thickness)
        image_drawn = cv2.putText(frame, Pred_name, text_point, font, fontScale, color, thickness, cv2.LINE_AA, False)


    t = time.time()-t0
    fps_array.append(1/t)
    fps_array[-150:]
    fps_average = sum(fps_array)/len(fps_array)

    cv2.imshow('VIDEO', frame)
    
    pressedKey = cv2.waitKey(1) & 0xFF

    if pressedKey == ord('q'):
        break
    
    print("AVERAGE FPS: " + str(fps_average))