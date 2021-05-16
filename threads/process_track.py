import cv2
import numpy as np 


max_distance_between_points = 30

def euclidean_distance(detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

def draw_border(img, box, line_length = 15):

        x1, y1 = box[0], box[1]
        x2, y2 = box[0], box[3]
        x3, y3 = box[2], box[1]
        x4, y4 = box[2], box[3]    
        
        cv2.line(img, (x1, y1), (x1 , y1 + line_length), (127, 0, 127), 5)  #-- top-left
        cv2.line(img, (x1, y1), (x1 + line_length , y1), (127, 0, 127), 5)

        cv2.line(img, (x2, y2), (x2 , y2 - line_length), (127, 0, 127), 5)  #-- bottom-left
        cv2.line(img, (x2, y2), (x2 + line_length , y2), (127, 0, 127), 5)

        cv2.line(img, (x3, y3), (x3 - line_length, y3), (127, 0, 127), 5)  #-- top-right
        cv2.line(img, (x3, y3), (x3, y3 + line_length), (127, 0, 127), 5)

        cv2.line(img, (x4, y4), (x4 , y4 - line_length), (127, 0, 127), 5)  #-- bottom-right
        cv2.line(img, (x4, y4), (x4 - line_length , y4), (127, 0, 127), 5)

        sub_img = img[box[1]: box[3], box[0]: box[2], :]
        white_rect = np.zeros(sub_img.shape, dtype=np.uint8) + 255
        res = cv2.addWeighted(sub_img, 0.7, white_rect, 0.3, 1.0)
        img[box[1]: box[3], box[0]: box[2], :] = res
        return img

def get_center(bbox):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def get_centroid(yolo_box, img_height, img_width):
        x1 = yolo_box[0] * img_width
        y1 = yolo_box[1] * img_height
        x2 = yolo_box[2] * img_width
        y2 = yolo_box[3] * img_height
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])