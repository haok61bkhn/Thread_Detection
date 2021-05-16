import os
import sys
sys.path.insert(0, "../yolov5")
import cv2
import numpy as np
# from utils_process.fps import FPS
import time
from predict import Detector
from queue import Queue
from threading import Thread
# from multiprocessing import Pool, Queue
# from decord import VideoReader
import norfair
from norfair import Detection, Tracker, Video
max_distance_between_points = 30

detector = Detector()

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

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

  # set use_cuda=False if using CPU

# for input_path in args.files


def worker_detect(input_q, detect_q):
    while True:
        frame = input_q.get()
        detect_q.put(detect_objects(frame))

def worker_tracking(detect_q, tracker_q):
    while True:
        box_detects = detect_q.get()
        frame = input_q.get()
        detections = [
            Detection(get_center(box), data=box)
            for box in box_detects
        ]
        tracked_objects = tracker.update(detections=detections)
        # norfair.draw_boxes(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        tracker_q.put(frame)

def detect_objects(image_np):
    boxes, _, _ = detector.detect(image_np)
    return boxes


if __name__ == '__main__':
    path_video = '/home/sonnh/Downloads/Counter_motpy/town.avi'
    # video_capture = cv2.VideoCapture(path_video)
    video = Video(input_path = path_video)
    font = cv2.FONT_HERSHEY_SIMPLEX
    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    detect_q = Queue()
    tracker_q = Queue()

    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )

    t_detect = Thread(target=worker_detect, args=(input_q, detect_q))
    t_tracking = Thread(target=worker_tracking, args=(detect_q, tracker_q))
    t_detect.daemon = True
    t_detect.start()
    t_tracking.daemon = True
    t_tracking.start()
    # pool_detect = Pool()

    i = -1
    for frame in video:
        i+=1
        if i % 2 ==0:
            input_q.put(frame)
            t = time.time()
            if tracker_q.empty():
                pass  # fill up queue
            else:
                frame = tracker_q.get()
                video.write(frame)
                # cv2.imshow('Video', frame)
            # print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    # video_capture.release()
    # cv2.destroyAllWindows()



# frame_num = 0

# for frame in video:
#     # frame_num += 1
#     if frame_num %3 ==0:
#         frame = np.array(frame)
#         box_detects, _, _ = detector.detect(frame)
#         detections = [
#             Detection(get_center(box), data=box)
#             for box in box_detects
#         ]
#         tracked_objects = tracker.update(detections=detections)
#         norfair.draw_points(frame, detections)
#         norfair.draw_tracked_objects(frame, tracked_objects)
#         video.write(frame)
        
        