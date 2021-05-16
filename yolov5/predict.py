import os
import platform
import shutil
import time
from pathlib import Path
from utils.datasets import  letterbox, letterbox_multi
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import glob
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized
from config import get_config
from PIL import Image
import numpy as np

# from random_object_id import generate

class Detector(object):

    def __init__(self, classes = []):
        opt = get_config()
        self.img_size =opt.img_size
        weights= opt.weights
        self.device = opt.device
        self.model = attempt_load(weights, map_location=self.device)
        self.conf_thres=opt.conf_thres
        self.iou_thres=opt.iou_thres
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if len(classes) == 0:
            self.names = self.classes
        else :
            self.names = classes

        print(self.names)


    def detect(self,im0s,img):
        img = letterbox(im0s, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img) 
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        box_detects=[]
        ims=[]
        classes=[]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *x, conf, cls in reversed(det):
                    if self.classes[int(cls)] in self.names :
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        ims.append(im0s[c1[1]:c2[1],c1[0]:c2[0]])
                        top=c1[1]
                        left=c1[0]
                        right=c2[0]
                        bottom=c2[1]
                        box_detects.append(np.array([left,top, right,bottom]))
                        classes.append(self.classes[int(cls)])
                    
        return box_detects,ims,classes

    def draw_bbox(self, img):
        t1 = time.time()
        boxes, ims, classes = self.detect(img)
        # print(time.time() - t1, len(boxes))
        font = cv2.FONT_HERSHEY_SIMPLEX 

        for box, im, lb in zip(boxes, ims, classes):
            cv2.imwrite("object/" + str(generate()) + ".jpg", im)
            img =cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,255,0), 1)
            cv2.putText(img, lb,(box[0],box[1]), font , 1, (255,0,0), 1 )

        return img

    def test_video(self, path):
        cap = cv2.VideoCapture(path)

        while(True):
            _, frame = cap.read()
            frame = self.draw_bbox(frame)
            frame = cv2.resize(frame, (1600, 900))
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    #---------------------------------Process synopsis video ---------------------------------------------------#
    def detect_multi(self, img0s,imgs) :
        img = torch.from_numpy(imgs).to(self.device).float()
        print("shape ",imgs.shape)
        img /= 255.0  
        
        pred = self.model(img, augment=False)[0]
        res=[]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        
        for i, det in enumerate(pred):  # 
            res_cur={}
            for cl in self.classes:
                res_cur[cl]=[]

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0s[0].shape[:2]).round()
                for *x, conf, cls in reversed(det):
                    if self.classes[int(cls)] in self.names :
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        top=c1[1]
                        left=c1[0]
                        right=c2[0]
                        bottom=c2[1]
                        res_cur[self.classes[int(cls)]].append([(left+right)//2,(top+bottom)//2,right-left,bottom-top]) #x y w h
            res.append(res_cur)
        return res
        
    def detect_multi2(self, imgs) :
        img_shape = imgs[0].shape
        img = letterbox_multi(imgs, new_shape=self.img_size)[0]
        im0s = np.ascontiguousarray(img) 
        img = torch.from_numpy(im0s).to(self.device).float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1=time.time()
        pred = self.model(img, augment=False)[0]
        res=[]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)
        t1=time.time()

        for i, det in enumerate(pred):  # 
            res_cur=[]

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_shape).round()
                for *x, conf, cls in reversed(det):
                    if self.classes[int(cls)] in self.names :
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        top=c1[1]
                        left=c1[0]
                        right=c2[0]
                        bottom=c2[1]
                        res_cur.append(np.array([left,top, right,bottom])) #x y w h
            res.append(res_cur)

        return res

if __name__ == '__main__':

    detector=Detector(classes = [])

    # detector.test_video('/home/nms173341/Documents/AI/Tracking/video_vn/town.avi')
    list_img = []

    t = time.time()
    for path_img in glob.glob('/home/nms173341/Downloads/test/*.jpg') :
        img = cv2.imread(path_img)
        list_img.append(img)
    det = detector.detect_multi(list_img)
    t1 = time.time()
    print(t1 - t)

    t12 = time.time()
    for path_img in glob.glob('/home/nms173341/Downloads/test/*.jpg') :
        img = cv2.imread(path_img)
        # t = time.time()
        det = detector.detect(img)
    # det = detector.detect(img)
    print(time.time() - t12)
    


