import os
import sys
sys.path.insert(0, "yolov5")
from predict import Detector
from multiprocessing import Queue
from norfair import  Video
from threads.read_thread import ReadThread
from threads.detect_thread import DetectThread
from threads.track_thread import TrackThread
from threads.prepare_image import PrepareImage
import time

if __name__ == '__main__':
   
    input_q = Queue(50) 
    input_det_q=Queue(50)
    detect_q = Queue()
    tracker_q = Queue()
    stt_q = Queue()
    video = Video(input_path = 'town.avi')

    
    thread_read = ReadThread(1, input_q, tracker_q, video, stt_q)
    thread_prepares=[PrepareImage(2,input_q,input_det_q) for i in range(3)]
    thread_detect = DetectThread(3, input_q, detect_q, Detector(),input_det_q)
    thread_detect1 = DetectThread(3, input_q, detect_q, Detector(),input_det_q)
    # thread_track = TrackThread(3, detect_q, tracker_q, stt_q)
    

    thread_read.start()
    for thread_prepare in thread_prepares:
        thread_prepare.start()
    thread_detect.start()
    thread_detect1.start()
   
    # thread_track.start()
    
    thread_read.join()
    del thread_read
    thread_prepare.lock=True
    thread_prepare.join()
    del thread_prepare
    thread_detect.lock=True
    thread_detect.join()
    thread_detect1.lock=True
    thread_detect1.join()