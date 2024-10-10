from detection.utils import iou

class ObjectReID():
    """
    Object ReID class, assume that no more ids are added after the initialization
    """

    def __init__(self):
        """
        Initialize the object reid class
        """
        self.id_list = []
        self.missing_ids = []
        self.id_bbox = []
        self.id_limit = 0

    def initialize_id_list(self, bbox_list):
        """
        Initialize the id list with the given bounding boxes
        Args:
            bbox_list (list): list of bounding boxes, form of [x1, y1, x2, y2, conf, id]
        """ 

        # Remove the bounding boxes with confidence less than 0.5

        rm_list = []
        for i in range(0, len(bbox_list)):
            if bbox_list[i][4] < 0.5:
                rm_list.append(bbox_list[i])

        for box in rm_list:
            bbox_list.remove(box)

        # Check if the confidence is greater than a threshold
        for i in range(0, len(bbox_list)):
            tmp_box = bbox_list[i]
            tmp_box[5] = i
            self.id_bbox.append(tmp_box)

        print("ID BOX", self.id_bbox)


    def update_id_list(self, bbox_list, state):
        """
        Update the id list with the new bounding boxes
        Args:
            bbox_list (list): list of bounding boxes
            state (str): state of the object reid, either "+" or "-"
        """
        # If the id list is empty, initialize the id list
        if len(self.id_bbox) == 0:
            self.initialize_id_list(bbox_list)
            return
        
        # Check if the confidence is greater than a threshold
        rm_list = []
        for i in range(0, len(bbox_list)):
            if bbox_list[i][4] < 0.5:
                rm_list.append(bbox_list[i])

        for box in rm_list:
            bbox_list.remove(box)

        #self.update_bbox_pos(bbox_list)

        print("Bbox list", bbox_list)
        print("Id Box", self.id_bbox)

        if state == "-":
            self.get_missing_ids(bbox_list)
        elif state == '+':
            self.assign_changed_id(bbox_list)
  
        return
        
    

    def get_missing_ids(self, bbox_list):
        """
        Get the missing ids from current bounding box prediction
        Args:
            bbox_list (list): list of bounding boxes, form of [x1, y1, x2, y2, conf, id]
        Returns:
            list: list of missing ids
        """

        # Compare self.id_bbox and bbox_list
        rm_list = []
        for id_box in self.id_bbox:
            max_iou = 0
            max_id = -1
            for bbox in bbox_list:
                iou_score = iou(id_box[:4], bbox[:4])
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_id = bbox[5]
            if max_iou < 0.5:
                self.missing_ids.append(id_box)
                rm_list.append(id_box)


        for box in rm_list:
            self.id_bbox.remove(box)

        print("After remove : MISSING IDS", self.missing_ids)
        print("ID BOX", self.id_bbox)


    def assign_changed_id(self, bbox_list):
        """
        Assign the lost id to the missing bounding boxes
        Args:
            bbox_list (list): list of bounding boxes, form of [x1, y1, x2, y2, conf, id]
        Returns:    
            list: list of bounding boxes with assigned ids
        """

        # Check between self.id_bbox and bbox_list, if there a new bbox in bbox_list, assign the missing_id to it and remove the missing_id from missing_ids

        if len(self.missing_ids) > 0:
            for bbox in bbox_list:
                max_iou = 0
                max_id = -1
                for id_box in self.id_bbox:
                    iou_score = iou(bbox[:4], id_box[:4])
                    if iou_score > max_iou:
                        max_iou = iou_score
                        max_id = id_box[5]
                if max_iou < 0.5:
                    bbox[5] = self.missing_ids[0][5]
                    self.missing_ids.remove(self.missing_ids[0])
                    self.id_bbox.append(bbox)

            print("After assign, ID BOX", self.id_bbox)
            print("MISSING IDS", self.missing_ids)  

        else:
            # When there are new objects, assign a new id to them
            for bbox in bbox_list:
                max_iou = 0
                for id_box in self.id_bbox:
                    iou_score = iou(bbox[:4], id_box[:4])
                    if iou_score > max_iou:
                        max_iou = iou_score
                if max_iou < 0.5:
                    bbox[5] = len(self.id_bbox)
                    self.id_bbox.append(bbox)

    def update_bbox_pos(self, bbox_list):
        """
        Update the bounding box position in the id list
        Args:
            bbox_list (list): list of bounding boxes, form of [x1, y1, x2, y2, conf, id]
        """
        # Update the bounding box position in the id list by comparing the IOU score with the predicted bounding boxes
        for bbox in bbox_list:
            max_iou = 0
            max_id = -1
            for id_bbox in self.id_bbox:
                iou_score = iou(bbox[:4], id_bbox[:4])
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_id = id_bbox[5]
                    max_box = id_bbox
            if max_iou > 0.1:
                bbox[5] = max_id
                self.id_bbox.remove(max_box)
                self.id_bbox.append(bbox)
    

    