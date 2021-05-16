import threading
import time

class ReadThread(threading.Thread):
    def __init__(self, thread_id, input_queue, track_queue, video, stt_queue,daemon=True):
        threading.Thread.__init__(self,daemon=daemon)
        self.thread_id = thread_id
        self.input_queue = input_queue
        self.track_queue = track_queue
        self.video = video
        self.stt_queue = stt_queue
        
    def run(self):
        print("Thread read and write video start")
        t = time.time()
        i = 0
        for frame in self.video:
           
            i+=1
            if i % 2 :
                self.input_queue.put(frame)
                # print("len queue",self.input_queue.qsize())
                # if not self.track_queue.empty():
                #     frame = self.track_queue.get()
                #     self.video.write(frame)
                    
        self.stt_queue.put(True)
        t2 = time.time() - t
        print("Time total : " + str(t2))