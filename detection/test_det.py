from ultralytics import YOLO
import torch
import os
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = True

# Initialize a YOLO-World model
# model = YOLO("yolov8s-world.pt")
# model = YOLO("yolov8s-worldv2.pt")
model = YOLO("/data1.local/vinhpt/dupt/factory/src/weights/yolo_world/yolov8x-worldv2.pt")


# Define custom classes
# model.set_classes(["person", "bus"])
model.set_classes(["person", "scanner", "basket", "mask", "pen", "glasses", "glove", "shoe", "boot", "clothes`"])

# Execute prediction for specified categories on an image
image_folder = "/data1.local/vinhpt/dupt/factory/image_test2/Blurry"
image_path_list = os.listdir(image_folder)
for image_path in image_path_list:
    img = cv2.imread(os.path.join(image_folder, image_path))
    results = model(img, save=True, conf=0.05, device=device, half=half, line_width=1)


# với từng class:
# Kết quả có đang tốt không? từng trường hợp (nét, mờ, thực tế)
# Confidence score hợp lý cho từng class?
# tốc độ chạy bao lâu
# thứ 3
