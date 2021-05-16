import os
import sys
sys.path.insert(0, "yolov5")
import argparse
import cv2
import numpy as np
import torch
from yolov5.predict import Detector
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

# for input_path in args.files:
video = Video(input_path = '/home/sonnh/Downloads/town_cut.mp4', output_fps = 30.0)
tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=max_distance_between_points,
)

frame_num = -1

for frame in video:
    frame_num += 1
    if frame_num % 2 ==0:
        frame = np.array(frame)
        box_detects, _, _ = detector.detect(frame)
        detections = [
            Detection(get_center(box), data=box)
            for box in box_detects
        ]
        tracked_objects = tracker.update(detections=detections)
        norfair.draw_points(frame, detections)
        norfair.draw_tracked_objects(frame, tracked_objects)
        video.write(frame)
        
        