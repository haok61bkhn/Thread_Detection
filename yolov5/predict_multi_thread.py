import numpy as np
import cv2
import os
import sys
from predict import Detector
import threading

detector = Detector()

class OutputFrame:
    def __init__(self, height, width):
        self.frame = np.zeros((height, width, 3))
        self.boxes = []

class WebcamThread(threading.Thread):
   def __init__(self, name):
      threading.Thread.__init__(self)
      self.name = name
   def run(self):
      print("Starting " + self.name)
      get_frame(self.name)
      print("Exiting " + self.name)

def get_frame(threadName):
    _, frame = cap.read()
    print(frame.shape)
    while frame is not None:
        _, frame = cap.read()
        output_frame.frame = frame

class PredictorThread(threading.Thread):
   def __init__(self, name):
      threading.Thread.__init__(self)
      self.name = name
   def run(self):
      print("Starting " + self.name)
      predict(self.name)
      print("Exiting " + self.name)

def predict(threadName):
    _, frame = cap.read()
    while frame is not None:
        _, frame = cap.read()
        boxes, _, _ = detector.detect(frame)
        output_frame.boxes = boxes
        


if __name__ == "__main__":

    path_video = "/home/nms173341/Documents/AI/Tracking/video_vn/town.avi"
    cap = cv2.VideoCapture(path_video)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    height, width = int(cap.get(3)), int(cap.get(4))
    print(height, width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # note the lower case
    # out = cv2.VideoWriter('output.avi', fourcc, 30, (width, height), True)
    
    output_frame = OutputFrame(height, width)

    # webcam_thread = WebcamThread(pa\th_video)
    predictor_thread = PredictorThread(path_video)
    # webcam_thread.start()
    predictor_thread.start()

    while True:
        if output_frame.boxes == []:
            to_show = output_frame.frame
        else:
            boxes = output_frame.boxes 
            to_show = output_frame.frame
            for box in boxes:
                img =cv2.rectangle(to_show, (box[0],box[1]), (box[2],box[3]), (0,255,0), 1)

        cv2.imshow('frame', to_show)
        # out.write(to_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv2.destroyAllWindows()