from roboflow import Roboflow
from PIL import Image
import cv2
import os, glob
import numpy as np

local_inference_server_address = "http://localhost:9001/"

rf = Roboflow(api_key="API")
project = rf.workspace().project("doors-and-windows-stage-1")
model = project.version(4, local=local_inference_server_address).model

rf_color = Roboflow(api_key="API")
project_color = rf_color.workspace().project("object-condition-stage-2")
model_color = project_color.version(1, local=local_inference_server_address).model

font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (255, 255, 255)
font_thickness = 3 
font_scale = 2

box_color = (255, 255, 255)
box_thickness = 3 
box_scale = 2

distance_color = (255, 255, 255)
distance_thickness = 3

# loop through folder of images
file_path = "test_images/" # folder of images to test - saved to output
# file_path = "images/single_image" # single image - saved to single_output
extention = ".jpg"
globbed_files = sorted(glob.glob(file_path + '*' + extention))
print(globbed_files)

counter = 0

for image_path in globbed_files:

    image_clean = cv2.imread(image_path)
    blk = np.zeros(image_clean.shape, np.uint8)

    pixel_ratio_array = []

    # infer on a local image
    response_json = model.predict(image_path, confidence=30, overlap=30).json()
    # print(response_json)

    for object in response_json["predictions"]:
        
        object_class = str(object['class'])
        object_class_text_size = cv2.getTextSize(object_class, font, font_scale, font_thickness)
        print("Class: " + object_class)
        object_confidence = str(round(object['confidence']*100 , 2)) + "%"
        print("Confidence: " + object_confidence)

        # pull bbox coordinate points
        x0 = object['x'] - object['width'] / 2
        y0 = object['y'] - object['height'] / 2
        x1 = object['x'] + object['width'] / 2
        y1 = object['y'] + object['height'] / 2
        box = (x0, y0, x1, y1)
        # print("Bounding Box Cordinates:" + str(box))

        image_cropped = image_clean[int(y0):int(y1), int(x0):int(x1)]

        image_cropped_path = "cropped/cropped_" + object_class + "_" + str(counter) + ".jpg"

        cv2.imwrite(image_cropped_path, image_cropped)

        # infer on a local image
        response_color_json = model_color.predict(image_cropped_path).json()
        
        object_color = response_color_json['predictions'][0]['top']

        print("Condition:" + object_color)

        box_start_point = (int(x0), int(y0))
        class_font_start_point = (int(x0), int(y0)-10)
        confidence_font_start_point = (int(x0)+object_class_text_size[0][0], int(y0)-10)
        box_end_point = (int(x1), int(y1))

        # print(start_point)
        # print(end_point)

        image_drawn = cv2.rectangle(image_clean, box_start_point, box_end_point, box_color, box_thickness)

        image_drawn_blk = cv2.rectangle(blk, box_start_point, box_end_point, box_color, box_thickness)

        image_drawn = cv2.putText(image_drawn, object_class, class_font_start_point, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        image_drawn = cv2.putText(image_drawn, " - " + object_confidence, confidence_font_start_point, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    cv2.imwrite("boxes" + str(counter) + ".jpg", image_drawn)

    counter += 1

cv2.waitKey(0)