from detection.utils import iou

def match_id(pred_bbox, id_box):
    """
    Match the predicted bounding boxes with the previous bounding boxes, and get its id
    Args:
        pred_bbox (list): list of predicted bounding boxes, form of [x1, y1, x2, y2, conf, id]
        id_box (list): list of previous bounding boxes, form of [x1, y1, x2, y2, conf, id]
    Returns:
        list: list of matched bounding boxes
    """

    
    matched_bbox = []
    for bbox in pred_bbox:
        max_iou = 0
        max_id = -1
        for id_bbox in id_box:
            iou_score = iou(bbox[:4], id_bbox[:4])
            if iou_score > max_iou:
                max_iou = iou_score
                max_id = id_bbox[5]
        if max_iou > 0.5:
            bbox[5] = max_id
            matched_bbox.append(bbox)
    return matched_bbox
