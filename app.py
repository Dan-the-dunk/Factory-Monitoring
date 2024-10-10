import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from face_reg.utils import update_database, get_face_detector_path, detect_face, iou_ellipse
from face_reg.retrieve import RecognizeFace
from face_reg.worker import Worker

from ultralytics import YOLO
import torch

#def register_face():

      
registering_list = ['Move your face to the elipse', 'Smile', 'Raise eyebrow', 'Squint your eye', 'Open your eye wide', 'Almost done']      
conf_dict = {"person" : 0.6, "scanner": 0, "basket" : 0.12, "mask": 0.06, "pen" :0.3, "glasses":0.5, "glove": 0.21, "shoe": 0.14, "boot": 0.13, "clothes" : 0.06}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
half = True


def main():
  pipeline = rs.pipeline()
  pipeline.start()

  flag = 'none'
  
  db_path = "face_reg/database/VN_Pruned_extracted"
  face_reg = RecognizeFace(db_path=db_path, model_name='Facenet512', enforce_detection=False)
  face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

  # Run once before the app camera starts
  face_reg.recognize_face("face_reg/database/VN_Pruned_extracted/455/0.png", silent=True)
  worker = Worker()

  model = YOLO("track_sm_3.9/detection/weights/yolov8x-worldv2.pt")
  class_list = ["person", "scanner", "basket", "mask", "pen", "glasses", "glove", "shoe", "boot", "clothes"]
  model.set_classes(class_list) 

  while True:
    
    frames = pipeline.wait_for_frames()
    
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    
    #org_frame = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    org_img = color_image.copy()

    key = cv2.waitKey(1) & 0xFF  # Get the key pressed by the user

    if key == 113:  # If the user pressed 'q' then quit the program
      break
    elif key == 115 and flag != 'save':  # If the user pressed 's' then save the image
      flag = 'save'
      new_faces = []
      f_start_time = time.time()
      save_start_time = time.time()
      print("Save started")
    elif key == 114 and flag == 'none':  # If the user pressed 'r' then start recognizing the face
      flag = 'recognize'
      print("Recognize started")

    # If the user pressed 's' and the time is less than 3 seconds take a new image every 0.6 seconds
    if flag == 'save' and time.time() - f_start_time < 18:
      # Put the time count to the frame
      if time.time() - save_start_time >= 3:
        new_faces.append(org_img)
        save_start_time = time.time()
        print("New image taken")
      
      # Put instruction to the screen, change every 3 second and add a 3 second count down
      cv2.putText(color_image, registering_list[int((time.time() - f_start_time) // 3)] + " :" + str(3 - int((time.time() - f_start_time) % 3)), (120 , 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
      # Draw an elipse at the middle of the screen 
      cv2.ellipse(color_image, (320, 240), (90, 110), 0, 0, 360, (0, 255, 255), 1)
      

    # When the timer is greater than 3 seconds, save the images
    if flag == 'save' and time.time() - f_start_time > 18:
      print(f"Save finished {time.time() - f_start_time}")
      
      # Count the number of folder in the database path and create a new folder with the next number
      folder_count = len([name for name in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, name))])
      
      for i, face in enumerate(new_faces):
        if not os.path.exists(f"face_reg/database/VN_Pruned_extracted/New User {folder_count}"):
          os.mkdir(f"face_reg/database/VN_Pruned_extracted/New User {folder_count}")
        cv2.imwrite(f"face_reg/database/VN_Pruned_extracted/New User {folder_count}/face_{i}.jpg", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        
        print("Start updating the database")
        update_database(db_path="face_reg/database/VN_Pruned_extracted", model_name='Facenet512', enforce_detection=False)
      
      new_faces = []
      flag = 'none' 

    if flag == 'recognize':
      color_image, faces = detect_face(color_image, face_det)
      # Put instruction to the screen
      cv2.putText(color_image, "Press 'esc' to stop", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
      if len(faces) > 0:
        worker.timer = time.time()
        if len(worker.id_list) < 5:
          results = face_reg.recognize_face(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
          print(results)
          for r in results:
            if not r.empty:
                # Extract the name
                name = r['identity'][0].split('/')[3]
                worker.id = name
                worker.id_list.append(name)
            else:
                print("No face found")
      # If no face seen in 3 seconds then reset the id_list
      if time.time() - worker.timer > 3:
        worker.id_list = []
      if key == 27: # If pressed 'esc' then stop recognizing
        flag = 'none'
        worker.id_list = []  

    # Write instruction to use the app in 'None' mode
    if flag == 'none':
      cv2.putText(color_image, "Press 's' to start register new face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
      cv2.putText(color_image, "Press 'r' to start recognize mode", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    if len(worker.id_list) >= 5 and flag == 'recognize':
      cv2.putText(color_image, worker.get_majority(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
      cv2.putText(color_image, "Show objects to camera", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

      

    
    # Object detection
    results = model.predict(source=org_img, conf=0.05, device=device, half=half, line_width=1, verbose=False)
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
        mask = torch.tensor([i not in rows_to_remove for i in range(bbox.shape[0])], dtype=torch.bool)
        bbox = bbox[mask]
        result.update(boxes=bbox)
        color_image = result.plot()
    cv2.imshow("DP" , cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    
  
if __name__ == "__main__":
  main()  