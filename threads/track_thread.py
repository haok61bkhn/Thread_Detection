import threading
import norfair
from norfair import Detection, Tracker, Video
import time
from .process_track import *

class TrackThread(threading.Thread):
    def __init__(self, thread_id, detect_queue, track_queue, stt_queue,daemon=True):
        threading.Thread.__init__(self,daemon=daemon)
        self.thread_id = thread_id
        self.detect_queue = detect_queue
        self.track_queue = track_queue
        self.tracker = Tracker(
            initialization_delay=0,
            distance_function=euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        self.stt_queue = stt_queue
        
    def run(self):
        print("Thread tracking start")

        while self.stt_queue.empty():
            box_detects, frame = self.detect_queue.get()
            detections = [
                Detection(get_center(box), data=box)
                for box in box_detects
            ]
            tracked_objects = self.tracker.update(detections=detections)
            for box in box_detects:
                draw_border(frame, box)

            norfair.draw_tracked_objects(frame, tracked_objects)
            self.track_queue.put(frame)

