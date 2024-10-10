from deepface import DeepFace
import time

img = 'face_reg/d1.jpg'
db_path = 'face_reg/VN_Pruned'
results = DeepFace.find(img, db_path=db_path, model_name='Facenet512', enforce_detection=False)

print(results)