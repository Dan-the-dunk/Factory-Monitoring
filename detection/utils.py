from ultralytics import YOLO
import json
import torch
import cv2


def is_intersected(box1, box2):
    """
    Check if two boxes are intersected
    Args:
        box1 (list): [x1, y1, x2, y2]
        box2 (list): [x1, y1, x2, y2]
    Returns:
        bool: True if two boxes are intersected, False otherwise
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    if x1 > x4 or x3 > x2:
        return False
    if y1 > y4 or y3 > y2:
        return False
    return True


def get_recent_item(items, threshold=2/3):
    """
    Get the most recent items from a list in a fraction (2/3) of the list
    Args:
        items (list): list of items, which is a list. EG: [[1,2,3], [1,3]]
        threshold (float): the fraction of the list to get the most recent items. EG: 2/3
    Returns:
        item (dict): the most recent item, also count the number of each class of item. EG {1: 2, 2: 3, 3: 2}
    """
    if len(items) <= 4:
        return []

    recent_items = items[-int(len(items) * threshold):]
    #print(recent_items)
    #print(len(recent_items))

    class_count = {}
    for item in recent_items:
        for sub_item in set(item):
            if sub_item in class_count:
                class_count[sub_item] += 1
            else:
                class_count[sub_item] = 1
    
    # Apply threshold for class_count
    item_cls = []
    for key, value in class_count.items():
        if value / len(recent_items) < 0.75:
            continue
        item_cls.append(key)    
    
    # Count the occurace of each item in the recentff_items
    item_count = []
    for item in recent_items:
        tmp_count = {}
        for item_sub in item:
            # If an item appear more than one time per sublist, use it occurance as key
            if item_sub in tmp_count:
                tmp_count[item_sub] += 1
            else:
                tmp_count[item_sub] = 1
        item_count.append(tmp_count)
    

    freq_dict = {}
    # The form of item_count needed is (class, count):freq
    for i in range(len(item_count)):
        for key, value in item_count[i].items():
            if (key,value) in freq_dict:
                freq_dict[(key,value)] += 1
            else:
                freq_dict[(key,value)] = 1

            
    # Check if the frequency of each item is greater than 0.9, if yes, return the item and its number
    item_freq = []
    for key, value in freq_dict.items():
        if value / len(recent_items) < 0.9 and key[0] not in item_cls:
            continue
    
        item_freq.append(key)
    # Only take the first element for each class, eg: [(1, 2), (2, 2), (1, 3) ] -> [(1, 2), (2,2)]
    for item in item_freq:
        for item2 in item_freq:
            if item[0] == item2[0] and item[1] != item2[1]:
                item_freq.remove(item2)


    return item_freq


    

def read_last_line(file_path):
    """Reads the last line of a JSON file and parses it.

    Args:
    file_path: The path to the JSON file.

    Returns:
    The parsed JSON object from the last line.
    """

    # Check if the file is empty
    cnt_line = 0
    line = []

    with open(file_path, 'r') as f:
        for line in f:
            cnt_line += 1
            pass
    if cnt_line == 0:
        return None
    last_line = line
    last_object = json.loads(last_line)
    return last_object

def iou(bbox1, bbox2):
    """
    Calculate the Intersection Over Union
    Args:
        bbox1 (list): [x1, y1, x2, y2]
        bbox2 (list): [x1, y1, x2, y2]
    Returns:
        float: the value of IOU
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    if x5 > x6 or y5 > y6:
        return 0

    intersection = (x6 - x5) * (y6 - y5)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    return intersection / (area1 + area2 - intersection)

def iob(bbox1, bbox2):
    """
    Calculate the Intersection Over Bounding Box
    Args:
        bbox1 (list): [x1, y1, x2, y2]
        bbox2 (list): [x1, y1, x2, y2]
    Returns:
        float: the value of IOB
    """

    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    if x5 > x6 or y5 > y6:
        return 0

    intersection = (x6 - x5) * (y6 - y5)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    return intersection / min(area1, area2)

def filter_object(bboxes, fixed_areas ,size_threshold=15000, iob_threshsold=0.5):
    """
    Filter the bounding boxes that have either too small IOB with the fixed areas (Start, Scanner, Finished) or bigger than a threshold
    Args:
        bboxes (list): list of bounding boxes, each bounding box is a list [x1, y1, x2, y2, conf, cls]
        fixed_areas (list): list of fixed areas, each fixed area is a list [x1, y1, x2, y2]
        size_threshold (int): the threshold of the bounding box size
    Returns:
        list: list of bounding boxes that don't satisfy the condition
    """

    # Calculate the size threshold
    size_threshold = (fixed_areas[0][3] - fixed_areas[0][1]) * (fixed_areas[0][2] - fixed_areas[0][0]) / 5

    #print(size_threshold)
    filtered_bboxes = []
    filtered_bboxes.append([1, 0, 0, 0, 0, 0])
    for bbox in bboxes:
        #print("Bbox :", bbox)
        #print("Bbox size: ", (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > size_threshold:
            continue

        flag = False
        for fixed_area in fixed_areas:
            if iob(bbox[:4], fixed_area) > iob_threshsold:
                flag = True
                break
        
        if flag:
            filtered_bboxes.append(list(bbox.cpu().numpy()))
    

    iob_filtered_bboxes = []
    iob_filtered_bboxes.append([1, 0, 0, 0, 0, 0])

    # Iterete throught all the filtered box, if the iob is greater than a threshold, delete the smaller box
    for bbox in filtered_bboxes:
        flag_iob = True
        for bbox2 in filtered_bboxes:
            if is_same_bbox(bbox[:4], bbox2[:4]):
                #print("Found same object")
                continue

            if iob(bbox[:4], bbox2[:4]) > 0.95 and (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) > (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]):
                #print("Found iob with another object")
                flag_iob = False
                break
        if flag_iob:
            iob_filtered_bboxes.append(list(bbox))

    ## Delete the smaller object that has iob with another object > a threshold.
    

    return torch.tensor(iob_filtered_bboxes, dtype=torch.float32)


def is_same_bbox(bbox1, bbox2):
    """
    Check if two bounding boxes are the same
    Args:
        bbox1 (list): [x1, y1, x2, y2]
        bbox2 (list): [x1, y1, x2, y2]
    Returns:
        bool: True if two bounding boxes are the same, False otherwise
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    if x1 == x3 and y1 == y3 and x2 == x4 and y2 == y4:
        return True
    return False

def count_items(item_bboxes, area_bboxes, iob_threshold=0.8):
    """
    Count the number of items in the predefined bboxes
    Args:
        item_bboxes (list): list of item bounding boxes, each bounding box is a list [x1, y1, x2, y2, conf, cls]
        area_bboxes (list): list of bounding boxes, each bounding box is a list [x1, y1, x2, y2], Start, Scanner, Finished
        iob_threshold (float): the threshold of IOB
    Returns:
        list: the number of each item in the predefined bboxes, in the order of Start, Scanner, Finished boxes
    """

    item_count = []
    for area_bbox in area_bboxes:
        cnt = 0
        for item_bbox in item_bboxes:
            if iob(item_bbox[:4], area_bbox) > iob_threshold:
                cnt += 1
        item_count.append(cnt)


    return item_count
    


def visualize_bbox(img, bboxes):
    """
    Visualize the bounding boxes on the image, each bounding box have different color and id
    Args:
        img (numpy array): the image
        bboxes (list): list of bounding boxes, each bounding box is a list [x1, y1, x2, y2, conf, id]
    Returns:    
        img (numpy array): the image with bounding boxes
    """
    if len(bboxes) == 0:
        return img

    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, conf, id = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[0], 2)
        #img = cv2.putText(img, str(int(i)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2)
        # Show id and confidence score
        img = cv2.putText(img, "ID:" + str(id), (x1 - 30, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[2], 1)
        img = cv2.putText(img, str(round(conf, 2)), (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[0], 1)
    return img



