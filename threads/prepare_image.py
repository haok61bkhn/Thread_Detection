import threading
import time
import cv2
import numpy as np
import torch
class PrepareImage(threading.Thread):
    def __init__(self, thread_id,input_q,input_det_q,batch_size=16,image_size=640,daemon=True):
        threading.Thread.__init__(self,daemon=daemon)
        self.thread_id = thread_id
        self.input_q=input_q
        self.input_det_q=input_det_q
        self.image_size=(image_size,image_size)
        self.lock=False
        self.batch_size=batch_size
    
    def letterbox(self,img,  color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        shape=img.shape[:2]
        new_shape=self.image_size
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  
            ratio = r, r  
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  
        elif scaleFill: 
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0] 

        dw /= 2  
        dh /= 2

        if shape[::-1] != new_unpad:  
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
        return img

        
    def run(self):
        print("Thread PrePare start")
        while not self.lock or self.input_q.empty():
            imgs=[]
            img0s=[]
            for i in range(min(self.batch_size,self.input_q.qsize())):
                frame = self.input_q.get()
                img0s.append(frame)
                img = self.letterbox(frame)
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
                
                imgs.append(img)
            imgs = np.ascontiguousarray(imgs) 
            if(len(img0s)>0):
                 
                self.input_det_q.put((img0s,imgs))
        print("end Thread PreProce")

