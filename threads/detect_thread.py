# from multiprocessing import Process
from threading import Thread
import time

class DetectThread(Thread):
    def __init__(self, thread_id, input_queue, detect_queue, detector, input_det_q,daemon=True):
        Thread.__init__(self,daemon=daemon)
        self.thread_id = thread_id
        self.detect_queue = detect_queue
        self.input_queue = input_queue
        self.detector = detector
        self.input_det_q = input_det_q
        self.lock=False
        
    def detect_objects(self, img0s,imgs):
       
        res= self.detector.detect_multi(img0s,imgs)
        return res

    def run(self):
        print("Thread detect start")
        while not self.lock or not self.input_det_q.empty():
            # print("len input_det_q ",self.input_det_q.qsize())
            img0s,imgs = self.input_det_q.get()
            self.detect_queue.put(self.detect_objects(img0s,imgs))
        print("end Thread detect")
