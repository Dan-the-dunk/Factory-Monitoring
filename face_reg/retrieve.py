from deepface import DeepFace
from typing import Union
import numpy as np

class RecognizeFace():
    def __init__(
        self, db_path, 
        model_name='Facenet512', 
        detector_backend='opencv', 
        align=True, 
        normalization='base', 
        expand_percentage=0, 
        enforce_detection=False, 
        distance_metric='cosine', 
        threshold=None, 
        anti_spoofing=False):
        
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.align = align
        self.normalization = normalization
        self.expand_percentage = expand_percentage
        self.enforce_detection = enforce_detection
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.anti_spoofing = anti_spoofing

    def recognize_face(
        self, 
        img : Union[str, np.ndarray],
        silent=True):
        """
        Recognize the face in the image

        Args:
            img (str | ndarray): The path to the image or the image as a numpy array
        """

        # Find the faces in the frame
        results = DeepFace.find(img, db_path=self.db_path,
                                model_name=self.model_name, 
                                enforce_detection=self.enforce_detection,
                                detector_backend='yolov8',
                                silent=silent)
        return results


