import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer)#, set_logging)#plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def run_detect_or(model,img,device,im0):
    # Run inference
    #img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    names = model.module.names if hasattr(model, 'module') else model.names
    half =True
    img = torch.from_numpy(img).to(device)
      # run once
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    augment=False
    pred = model(img, augment=augment)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, 0.3, 0.5, classes=None, agnostic=False)
    # Process detections
    motor_class = []
    #bicycle_class = []
    car_class = []
    #motorbike_class = []
    bus_class = []
    truck_class = []
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
             # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):

                
             #   label = '%s %.2f' % (names[int(cls)], conf)
             #   plot_one_box(xyxy, im0, label=label, color=None, line_thickness=3)
                predicted_class = names[int(cls)]
                if predicted_class == 'motor' or\
                    predicted_class == 'bus' or\
                    predicted_class == 'car' or\
                    predicted_class == 'truck':
                    x = xyxy[0]
                    y = xyxy[1]
                    w = xyxy[2]-x
                    h = xyxy[3]-y
                    if predicted_class == 'motor':
                        motor_class.append([x, y, w, h, conf])
                        continue
                    if predicted_class == 'car':
                        car_class.append([x, y, w, h, conf])
                        continue
                    if predicted_class == 'bus':
                        bus_class.append([x, y, w, h, conf])
                        continue
                    if predicted_class == 'truck':
                        truck_class.append([x, y, w, h, conf])
                        continue
                    print(x,y,w,h)
                else:
                   continue
                        
            # Print results
    #dets=[car_class]        
    dets = [motor_class, car_class, bus_class, truck_class]

    return dets

def run_detect(model,img,device,im0):
    # Run inference
    #img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    names = model.module.names if hasattr(model, 'module') else model.names
    half = True
    img = torch.from_numpy(img).to(device)
      # run once
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    augment=False
    pred = model(img, augment=augment)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, 0.3, 0.5, classes=None, agnostic=False)
    # Process detections
    #bicycle_class = []
    cars = []
    trucks = []
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
             # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = 0
                
             #   label = '%s %.2f' % (names[int(cls)], conf)
             #   plot_one_box(xyxy, im0, label=label, color=None, line_thickness=3)
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


    return [cars, trucks]
