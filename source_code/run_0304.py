import argparse
import os
import platform
import shutil
import time
import numpy as np
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from models.experimental import *
from utils.datasets import *
from utils.general import *

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from object_tracking import OT

from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

begin_script = time.time()

data_path = '../Dataset_A'
list_video_path = '../Dataset_A/datasetA_vid_stats.txt'
id_path = '../Dataset_A/list_video_id.txt'
zones_path = './add/ROIs'
roex_path = './add/ROI_e'
video_path = '../Dataset_A'
result_path = './submission_output'
mois_path = './add/movement_description'
multi_path = './add/movement_multi'

filename_path = os.path.join(result_path, 'submission.txt')
result_file = open(filename_path, 'w')


def convert_multiPoint(MOI):
    sets = []
    for d, ps in MOI.items():
      a = []
      for p in ps: 
        a.append(Point(p))
      sets.append(a)
    return sets

def cut_roi(image, roi):
    mask_roi = np.zeros((image.shape), np.uint8)
    roi = np.array([roi], dtype=np.int32)
    mask_roi = cv2.fillPoly(mask_roi, roi, (255, 255, 255))
    #mask_roi = cv2.fillPoly(mask_roi, roi, (0, 0, 0))
    image = cv2.bitwise_and(image, mask_roi) 
    return image 
    
def draw_roi(roi, frame):
    roi_nums = len(roi)-1
    for i in range(roi_nums):
        if i < roi_nums-1:
            cv2.line(frame,roi[i],roi[i+1],(0,255,0),2)
        else:
            cv2.line(frame,roi[i],roi[0],(0,255,0),2)
    return frame

def load_roi_moi(rois_path, rois_ex_path, mois_path, multi_path, name_video):
    roi = []
    roi_ex = []
    mois = {}
    multi = {}
    list_moi_edge=[]
    cam_index = name_video.split('_')[1]
    if len(cam_index) > 2:
        cam_index = cam_index.split('.')[0]
    with open(os.path.join(rois_path, 'cam_{}.txt'.format(cam_index))) as f:
        for p in f:
            p = p.rstrip("\n")
            p = p.split(',')
            temp = p[2:]
            temp = [int(x) for x in temp]
            list_moi_edge.append(temp)
            roi.append((int(p[0]), int(p[1])))
    roi.append(list_moi_edge)
    
    with open(os.path.join(rois_ex_path, 'cam_{}.txt'.format(cam_index))) as f:
        for p in f:
            p = p.rstrip("\n")
            p = p.split(',')
            temp = p[2:]
            temp = [int(x) for x in temp]
            roi_ex.append((int(p[0]), int(p[1])))

    with open(os.path.join(multi_path, 'cam_{}.txt'.format(cam_index))) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if len(line) == 0: continue
            a = line.split(',')
            temp = []
            for ii in range(0,len(a)-1,2):
              temp.append((int(a[ii]), int(a[ii+1])))
            multi[i+1]=temp

    with open(os.path.join(mois_path, 'cam_{}.txt'.format(cam_index))) as f:
      for i, line in enumerate(f):
        line = line.rstrip("\n")
        if len(line) == 0: continue
        a = line.split(',')
        p1 = (int(a[0]),int(a[1]))
        p2 = (int(a[2]),int(a[3]))
        p3 = (int(a[4]),int(a[5]))
        p4 = (int(a[6]),int(a[7]))
        l1 = (int(a[8]),int(a[9]))
        l2 = (int(a[10]),int(a[11]))
        mois[i+1]=[p1,p2,p3,p4,l1,l2]

    return roi, roi_ex, mois, multi

def load_list_video(input_path, id_path):
    names = []
    ids = []
    info = []
    with open(id_path,'r') as f:
        for line in f:
            a = line.split(' ')
            ids.append(a[0])
            names.append(a[-1].split('\n')[0])

    with open(input_path,'r') as f:
        for line in f:
            video_name = line.split('\t')[0]

            try:
                fps = line.split('\t')[1]
                total_frame = int(line.split('\t')[2])
                if fps != 'fps':
                    fps = fps.split('/')[:-1]
                    fps = int(fps[0])
                id = ids[names.index(video_name)]
                info.append([id, video_name, fps, total_frame])
            except:
                pass
    #print(info)
    return info

def load_delay(path, name_video):
    f = open(path,'r').readlines()
    posi = 0
    for i, line in enumerate(f):
        line = line.rstrip("\n")
        if line == name_video:
            posi = i
            break
    car = [int(t) for t in f[posi+1].rstrip('\n').split(',')]
    truck = [int(t) for t in f[posi+2].rstrip('\n').split(',')]
    return [car, truck]

def write_result_file(data, name):
    i = 0
    c = 0
    e = 0
    while i <len(data):
      d_t = data[i]
      d_s = []
      if i < len(data)-1:
        d_s = data[i+1]
        if abs(d_t[0]-d_s[0]) <= 0.01 and d_t[2] == d_s[2] and d_t[3] == d_s[3] and d_t[4] != d_s[4]:
          d = d_t if (d_t[4] == 1) else d_s
          result_file.write('{} {} {} {} {}\n'.format(d[0], d[1], d[2], d[3], d[4]))
          e += 1
          c += 1
          i += 2
          continue
      d = d_t
      result_file.write('{} {} {} {} {}\n'.format(d[0], d[1], d[2], d[3], d[4]))
      c += 1
      i += 1

    print(name+' counted: {}, saved: {}, error: {}'.format(len(data), c, e))


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)
def detect(source, weights, imgsz, device, names, half, skip):
    
    names = load_classes(names)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    frame_num = 1
    skip_frame = 1
    frame_track = 1
    data = []
    counter = 0
    video_capture = cv2.VideoCapture(source)
    w = int(video_capture.get(3))
    h = int(video_capture.get(4))
    shape_img = (w,h)
    if writeVideo_flag:
      # Define the codec and create VideoWriter object
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')#cv2.VideoWriter_fourcc(*'MJPG')
      out = cv2.VideoWriter('result_'+info[1], fourcc, info[-2], (w, h))
    
    begin_time = time.time()
    while True:
      t= time.time()
      ret,frame = video_capture.read()
      if ret != True:
        break
      if skip_frame <= skip :
        frame_num +=1
        skip_frame +=1
        continue
      frame = cut_roi(frame,ROI_ex)
      img = letterbox(frame, new_shape=imgsz)[0]
      img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
      img = np.ascontiguousarray(img)
      res = frame.copy()

      img = torch.from_numpy(img).to(device)
      img = img.half() if half else img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
        img = img.unsqueeze(0)
      
      # Inference
      #t1 = time_synchronized()
      ta = time.time()
      pred = model(img, augment=False)[0]
      
      # Apply NMS
      pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
      #t2 = time_synchronized()

      # Apply Classifier
      cars = []
      trucks = []
      # Process detections
      for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
          # Rescale boxes from img_size to im0 size
          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
          # Write results
          for *xyxy, conf, cls in det:
            predicted_class = names[int(cls)]
            x = xyxy[0]
            y = xyxy[1]
            w = xyxy[2]-x
            h = xyxy[3]-y
            if predicted_class == 'car':
              cars.append([x, y, w, h, conf])
            elif predicted_class == 'truck':
              trucks.append([x,y,w,h,conf])
            else:
              continue

      dets = [cars, trucks]
      for det, ob in zip (dets, objects):
        ob.begin_time = begin_time
        ob.frame_track = frame_track#frame_num
        ob.shape_img = shape_img
        ob.predict_obtracker(res, det)
        ob.update_obtracker()
        if writeVideo_flag: res= ob.visualize(res)
        res, data = ob.tracking_ob1(res,data,frame_num,writeVideo_flag)

      if writeVideo_flag:
        #save a frame
        draw_roi(ROI, res)
        cv2.putText(res, str(frame_num)+'  '+str(len(data)),(50,50),0, 5e-3 * 300, (0,0,255),2)
        out.write(res)

      # if len(data) >= counter + 200:
      #   counter = len(data)
      #   print(info[1]+' frame_num: {} fps: {}  saved: {}'.format(frame_num, np.round(1/(time.time()-t),3), len(data)))
      #print(info[1]+' frame_num: {} fps: {}  saved: {}'.format(frame_num, np.round(1/(time.time()-t),3), len(data)))
      frame_num +=1
      frame_track +=1
      skip_frame = 1

    write_result_file(data, name=info[1])
    
if __name__ == '__main__':
    
    with torch.no_grad():

      imgsz = 512
      # Initialize
      device = select_device('')
      half = device.type != 'cpu'  # half precision only supported on CUDA
      # Load model
      weights = '../weights/best_yolov4_csp-3-0.25_512_sync.pt'
      cfg = '../weights/yolov4-csp-3-0.25.cfg'
      names = './data/coco.names'
      model = Darknet(cfg, imgsz).cuda()
      try:
          model.load_state_dict(torch.load(weights, map_location=device)['model'])
      except:
          model = model.to(device)
          load_darknet_weights(model, weights)
      model.to(device).eval()
      print(weights)
      print('imgsz', imgsz)
      print('half', half)
      if half:  model.half()  # to FP16

      counter = []
      writeVideo_flag = False

      info_cam = load_list_video(list_video_path, id_path)
      
      max_cosine_distance=0.8
      nn_budget = 100
      nms_max_overlap = 1.0
      time_offset = 0

      for info in info_cam: 
        path = os.path.join(video_path, info[1])
        ROI, ROI_ex, MOI, multi = load_roi_moi(zones_path, roex_path, mois_path, multi_path, info[1])
        setofpoint = convert_multiPoint(multi)
        frame_delay = load_delay('./add/Dataset_A/time_delay.txt', info[1])
        name = info[1].split('.')[0]
        print("Processing video: ", info)
        skip = 1 if (int(info[0]) in [6, 8, 11, 14, 21, 28]) else 2
        car_class = OT(1,'Car',setofpoint,multi,MOI,ROI,info,frame_delay)
        truck_class = OT(2,'Truck',setofpoint,multi,MOI,ROI,info,frame_delay)
        objects = [car_class, truck_class]
        
        detect(path, weights, imgsz, device, names, half, skip)
        del objects
        print('saved ')
      result_file.close()
      print('Done')
      print('total time: {} sec'.format(np.round(time.time()-begin_script,3)))
       
          
