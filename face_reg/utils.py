from typing import Optional
import numpy as np
import cv2
import pickle as pickle
import os
import time
from deepface.commons import image_utils
from deepface.commons import logger as log
from deepface.modules.recognition import __find_bulk_embeddings 


logger = log.get_singletonish_logger()


def update_database(
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    refresh_database: bool = True):
    """
    Update the database with the new images
    Args:
        db_path: Path to the database
        model_name: Model name
        distance_metric: Distance metric
        enforce_detection: Whether to enforce detection
        detector_backend: Detector backend
        align: Whether to align
        expand_percentage: Expand percentage
        normalization: Normalization
        silent: Whether to suppress logging
        refresh_database: Whether to refresh the database
    Returns:
        None
    """
    
    
    tic = time.time()
    
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    file_parts = [
        "ds",
        "model",
        model_name,
        "detector",
        detector_backend,
        "aligned" if align else "unaligned",
        "normalization",
        normalization,
        "expand",
        str(expand_percentage),
    ]

    file_name = "_".join(file_parts) + ".pkl"
    file_name = file_name.replace("-", "").lower()

    datastore_path = os.path.join(db_path, file_name)
    representations = []

    # required columns for representations
    df_cols = [
        "identity",
        "hash",
        "embedding",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    # Ensure the proper pickle file exists
    if not os.path.exists(datastore_path):
        with open(datastore_path, "wb") as f:
            pickle.dump([], f)

    # Load the representations from the pickle file
    with open(datastore_path, "rb") as f:
        representations = pickle.load(f)

    # check each item of representations list has required keys
    for i, current_representation in enumerate(representations):
        missing_keys = list(set(df_cols) - set(current_representation.keys()))
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )

    # embedded images
    pickled_images = [representation["identity"] for representation in representations]

    # Get the list of images on storage
    storage_images = image_utils.list_images(path=db_path)

    if len(storage_images) == 0 and refresh_database is True:
        raise ValueError(f"No item found in {db_path}")
    if len(representations) == 0 and refresh_database is False:
        raise ValueError(f"Nothing is found in {datastore_path}")

    must_save_pickle = False
    new_images = []
    old_images = []
    replaced_images = []

    if not refresh_database:
        logger.info(
            f"Could be some changes in {db_path} not tracked."
            "Set refresh_database to true to assure that any changes will be tracked."
        )

    # Enforce data consistency amongst on disk images and pickle file
    if refresh_database:
        new_images = list(set(storage_images) - set(pickled_images))  # images added to storage
        old_images = list(set(pickled_images) - set(storage_images))  # images removed from storage

        # detect replaced images
        for current_representation in representations:
            identity = current_representation["identity"]
            if identity in old_images:
                continue
            alpha_hash = current_representation["hash"]
            beta_hash = image_utils.find_image_hash(identity)
            if alpha_hash != beta_hash:
                logger.debug(f"Even though {identity} represented before, it's replaced later.")
                replaced_images.append(identity)

    if not silent and (len(new_images) > 0 or len(old_images) > 0 or len(replaced_images) > 0):
        logger.info(
            f"Found {len(new_images)} newly added image(s)"
            f", {len(old_images)} removed image(s)"
            f", {len(replaced_images)} replaced image(s)."
        )

    # append replaced images into both old and new images. these will be dropped and re-added.
    new_images = new_images + replaced_images
    old_images = old_images + replaced_images

    # remove old images first
    if len(old_images) > 0:
        representations = [rep for rep in representations if rep["identity"] not in old_images]
        must_save_pickle = True

    # find representations for new images
    if len(new_images) > 0:
        representations += __find_bulk_embeddings(
            employees=new_images,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
            normalization=normalization,
            silent=silent,
        )  # add new images
        must_save_pickle = True

    if must_save_pickle:
        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f)
        if not silent:
            logger.info(f"There are now {len(representations)} representations in {file_name}")

    # Should we have no representations bailout
    if len(representations) == 0:
        if not silent:
            toc = time.time()
            logger.info(f"find function duration {toc - tic} seconds")
        return []

def get_opencv_path() -> str:
    """
    Returns where opencv installed
    Returns:
        installation_path (str)
    """
    opencv_home = cv2.__file__
    folders = opencv_home.split(os.path.sep)[0:-1]

    path = folders[0]
    for folder in folders[1:]:
        path = path + "/" + folder

    return path + "/data/"

def get_face_detector_path():
    opencv_path = get_opencv_path()
    face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
    if os.path.isfile(face_detector_path) != True:
        raise ValueError(
            "Confirm that opencv is installed on your environment! Expected path ",
            face_detector_path,
            " violated.",
        )

def detect_face(img, face_detector):
    """
    Detect face(s) from a given image and draw it
    Args:
        img (np.ndarray): pre-loaded image
        face_detector: YOLO face detector
    Returns:
        faces (list): list of faces
    """
    
    #height, width, _ = img.shape

    results = face_detector.predict(img, verbose=False, stream=True)
    # Get the faces from the results
    
    for r in results:
        img = r.plot()   
        faces = r.boxes.xyxy     
    
    return img, faces



def iou_ellipse(bbox, ellipse):
    """
    Calculate the intersection over union between the bounding box and the ellipse
    Args:
        bbox: bounding box, format: (x, y, w, h)
        ellipse: ellipse, format (center_x, center_y, a, b)
    Returns:
        iou (float): intersection over union
    """
    x1, y1, w1, h1 = bbox
    x2, y2, a, b = ellipse

    # Calculate the area of the bounding box
    area_bbox = w1 * h1

    # Calculate the area of the ellipse
    area_ellipse = np.pi * a * b

    # Calculate the intersection of the bounding box and the ellipse
    x1 = max(x1, x2 - a)
    x2 = min(x1 + w1, x2 + a)
    y1 = max(y1, y2 - b)
    y2 = min(y1 + h1, y2 + b)

    # Calculate the area of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area_intersection = w * h

    # Calculate the union of the bounding box and the ellipse
    area_union = area_bbox + area_ellipse - area_intersection

    # Calculate the intersection over union
    iou = area_intersection / area_union

    return iou
