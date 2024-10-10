from ultralytics import FastSAM
import torch
import os
import cv2
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = True


# Initialize a YOLO-World model
# model = YOLO("yolov8s-world.pt")
model = FastSAM("track_sm_3.9/detection/weights/FastSAM-s.pt")
#model = YOLO("/data1.local/vinhpt/dupt/factory/src/weights/yolo_world/yolov8x-worldv2.pt")

# Define custom classes
# model.set_classes(["person", "bus"])

class_list = ["person", "scanner", "basket", "mask", "pen", "glasses", "glove", "shoe", "boot", "clothes"]
model.set_classes(class_list)

# Execute prediction for specified categories on an image
image_folder = "track_sm_3.9/detection/test_images/Blurry"
image_path_list = os.listdir(image_folder)
for image_path in image_path_list:
    img = cv2.imread(os.path.join(image_folder, image_path))
    results = model(img, conf=0.05, device=device, half=half, line_width=1, verbose=False)
    # Save the results to a folder
    for result in results:
        # Check if the folder is created
        if not os.path.exists(image_folder + "/results"):
            os.mkdir(image_folder + "/results")
        
        print(image_folder + "/results/" + image_path)
        result.save(filename=image_folder + "/results/" + image_path)
