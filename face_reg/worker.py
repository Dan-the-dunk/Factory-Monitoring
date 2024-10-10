import time
from detection.utils import count_items

class Worker:
    def __init__(self, id : str = 'None'):
        self.id = id
        self.id_list = []
        self.timer = 0

        self.items_limit = 0
        self.reg_timer = 0
        self.scanned_items = 0

        self.scanner_flag = True

        self.start_items = []
        self.finish_items = []
        self.scanner_items = []

    def get_majority(self):
        # Get the majority of the id_list
        if len(self.id_list) > 0:
            major_id = max(set(self.id_list), key = self.id_list.count)
            return major_id
        else:
            return None
        

    def update_item(self, bbox, area_boxes):
        """Update the items list history with the item, save only 20 frames
        Args:
            bbox (list): the bounding box of the item
            area_boxes (list): the bounding box of the predefined area
        Returns:
            None
        """
        if len(bbox) == 0:
            return
        if len(self.start_items) > 24:
            self.start_items.pop(0)
            self.scanner_items.pop(0)
            self.finish_items.pop(0)

        start_cnt, scanner_cnt, finish_cnt = count_items(item_bboxes=bbox, area_bboxes=area_boxes)

        self.start_items.append(start_cnt)
        self.scanner_items.append(scanner_cnt)
        self.finish_items.append(finish_cnt)

    def get_item_counts(self, frame_threshold = 2/3):
        """
        Get the number of item each predefined box which satisfies the frame threshold
        Args:
            frame_threshold (float) : The threshold of the history frame that the item must satisfy EG: 2/3
        Returns:
            items_count (list) : The number of item that sastisfies the frame threshold in each predefined box (start, scanner, finish)
        """

        if len(self.start_items) <= 20:
            return []


        #print("Start items: ", self.start_items)
        #print("Scanner items: ", self.scanner_items)
        #print("Finish items: ", self.finish_items)

        recent_items = []

        recent_items.append(self.start_items[-int(len(self.start_items) * frame_threshold):])
        recent_items.append(self.scanner_items[-int(len(self.scanner_items) * frame_threshold):])
        recent_items.append(self.finish_items[-int(len(self.finish_items) * frame_threshold):])

        final_items = []
        for item in recent_items:
            # Calculate the frequency of each item
            freq_dict = {}  
            for i in item:
                if i in freq_dict:
                    freq_dict[i] += 1
                else:
                    freq_dict[i] = 1
            
            #print("Freq dict :", freq_dict)
            # Check if the frequency of each item is greater than 0.5, if yes, return the item and its number

            # Return the item that has the highest frequency
            item_count = []
            if len(freq_dict) > 0:
                item_count = [max(freq_dict, key=freq_dict.get)]
            else:
                item_count = [0]

            """item_count = []
            for key, value in freq_dict.items():
                if value / len(item) < 0.5:
                    continue
                item_count.append(key)
            
            if len(item_count) == 0:
                item_count = [0]"""


            final_items.append(item_count[0])

        #print("Final item: ", final_items)

        return final_items
     
    def update_item_limit(self, limit):
        """
        Update the item limit
        Args:
            limit (int) : the number of items that the worker can carry
        Returns:
            None
        """
        if limit > self.items_limit:
            self.items_limit = limit

    def update_scanner_count(self):
        """
        Update the scanner count
        Args:
            None
        Returns:
            None
        """
        if self.scanner_flag == True:
            self.scanned_items += 1
            self.scanner_flag = False
        