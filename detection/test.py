from ultralytics import YOLO
import torch
import os
import cv2
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = True

conf_dict = {"person" : 0.6, "scanner": 0, "basket" : 0.12, "mask": 0.14, "pen" :0.25, "glasses":0.2, "glove": 0.21, "shoe": 0.14, "boot": 0.13, "clothes" : 0.2}

# Initialize a YOLO-World model
# model = YOLO("yolov8s-world.pt")
model = YOLO("track_sm_3.9/detection/weights/yolov8x-worldv2.pt")
#model = YOLO("/data1.local/vinhpt/dupt/factory/src/weights/yolo_world/yolov8x-worldv2.pt")

# Define custom classes
# model.set_classes(["person", "bus"])

class_list = ["person", "scanner", "basket", "mask", "pen", "glasses", "glove", "shoe", "boot", "clothes"]
model.set_classes(class_list)

# Execute prediction for specified categories on an image
"""image_folder = "track_sm_3.9/detection/test_images/Blurry"
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
        result.save(filename=image_folder + "/results/" + image_path)"""


"""results = model.predict(source='track_sm_3.9/detection/test_images/Real/Boot1.jpg', conf=0.05, device=device, half=half, line_width=1, verbose=False)
for result in results:
    conf = result.boxes.conf.tolist()
    cls = result.boxes.cls.int().tolist()
    bbox = result.boxes.data.cpu()
    rows_to_remove = []

    # If each predition is lower than the threshold, delete from the results
    for i in range(len(cls)):
        print(cls[i])
        if conf[i] < conf_dict[class_list[cls[i]]]:
            rows_to_remove.append(i)
    
    # Delete all the rows in bbox that have the index in del_box, shape of bbox is (n, 6)
    print("Delete box :", rows_to_remove)
    mask = torch.tensor([i not in rows_to_remove for i in range(bbox.shape[0])], dtype=torch.bool)
    bbox = bbox[mask]

    print("After shape :", bbox.shape)

    result.save(filename="track_sm_3.9/org_test_img.jpg")

    result.update(boxes=bbox)

    # Save the results to a folder  
    result.save(filename="track_sm_3.9/test_img.jpg")"""


# với từng class:
# Kết quả có đang tốt không? từng trường hợp (nét, mờ, thực tế)
# Confidence score hợp lý cho từng class?
# tốc độ chạy bao lâu
# thứ 3
