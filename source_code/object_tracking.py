from __future__ import division, print_function, absolute_import
import os
import datetime
import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections1 as gdet
from application_util import visualization
from deep_sort.detection import Detection as ddet
from collections import deque
from check_moi import*
import math
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

np.random.seed(1)
class OT(object):
    
    def __init__(self, class_id, names, setofpoint, multi, MOI, ROI, info,frame_delay):
        self.max_cosine_distance = 0.9
        self.nms_max_overlap = 0.5
        self.nn_budget = None
        self.model_filename = 'model_data/market1501.pb'
        self.detections = []
        self.id = class_id
        self.boxes_tracked = []
        self.color = (255, 255, 255)
        #self.encoder = gdet.create_box_encoder(self.model_filename,batch_size=1)
        self.class_names = names
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.obtracker = Tracker(metric, id_cam = int(info[0]))
        self.pts = [deque(maxlen=1000) for _ in range(999999)]
        #self.point_first = [deque(maxlen=2) for _ in range(999999)]
        self.vis=visualization.Visualization(img_shape=(960,1280,3), update_ms=2000)
        COLORS = np.random.randint(100, 255, size=(255, 3), dtype="uint8")
        self.color = [int(c) for c in COLORS[self.id]]
        self.setofpoint = setofpoint
        self.multi = multi
        self.MOI = MOI
        self.ROI = ROI
        self.frame_track = 1
        self.info = info
        self.frame_delay = frame_delay
        self.begin_time = 0
        self.shape_img = None
        #self.total_frame = total_frame
        
    def predict_obtracker(self, frame, dets):
        #tt = time.time()
        boxs = [d[:4] for d in dets]
        #features = gdet.HOG_feature(frame, boxs)
        #features = gdet.create_his(frame, boxs)
        self.detections = [Detection(det[:4], det[4], None) for det in dets]    
        self.obtracker.predict()
        #print('times', np.round(time.time()-tt, 5))
    
    def update_obtracker(self):
        self.obtracker.update(self.detections, self.frame_track)
    def remove_track(self, ids_del):
        for track in self.obtracker.tracks:
            if track.track_id == ids_del:
                self.obtracker.tracks.remove(track)
                break
    
    def tracking_ob1(self,frame,data,frame_id,draw):
        #print('number', len(self.obtracker.tracks))
        for track in self.obtracker.tracks:
            if (not track.is_confirmed()):# or not track.match):
                continue
            bbox = track.to_tlbr()
            color = self.color
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            vel = 10
            dx  =0
            dy = 0
            if len(self.pts[track.track_id]) > 0:
                pl = self.pts[track.track_id][-1]
                vel = math.sqrt((center[0]-pl[0])**2+(center[1]-pl[1])**2)
                if vel < 5: continue
            self.pts[track.track_id].append(center)
            #center point
            if draw: cv2.circle(frame, (center), 4,(0,0,255), -1)
            old = len(self.pts[track.track_id])
            #if frame_id > 10 and l < 5: continue
            p0 = self.pts[track.track_id][0]
            p1 = self.pts[track.track_id][int(old/3)]
            p2 = self.pts[track.track_id][int(old*2/3)]
            p3 = self.pts[track.track_id][old-1]
            vel = math.sqrt((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)

            if not track.out_roi and vel >= 10:
                try:
                    direc_proposed, index = predict_direction(self.ROI, [p0,p1,p2,p3], self.MOI, conf=2.2)
                except:
                    print('bug predict direc')
                    direc_proposed = None
                    pass
                #direc_proposed, index = predict_direction(self.ROI, [p0,p1,p2,p3], self.MOI, conf=2.4)

                if direc_proposed != None:
                    points = list(self.pts[track.track_id])
                    points = points[::2 if (old >=2) else old]
                    #points = points[:int(old/(2 if (old >=2) else old) )]
                    #points = points[:int(old/(2 if (old >=2) else old) ):2 if (old >=2) else old ]
                    direc = predict_direction_nearest_v2(self.multi, direc_proposed,points, self.setofpoint)
                    track.out_roi = True
                    if direc != 0:
                        delay = self.frame_delay[int(self.id)-1][int(direc-1)]
                        timestamp = frame_id
                        if delay == 1:
                            if draw: frame = self.draw_direc(index, direc, frame)
                            timestamp += 0
                            if timestamp > self.info[-1]: timestamp = self.info[-1]
                            sec = np.round(time.time()-self.begin_time, 2)
                            data.append([sec,self.info[0], timestamp, direc, self.id])
                            #print([sec,self.info[0], timestamp, direc, self.id])
                            #self.obtracker.tracks.remove(track)
                        else:
                            if draw: frame = self.draw_direc(index, 'w', frame)
                            track.wait_loss = True
                            track.direc_ = direc
                            track.posi_cut = index
                            #pl = self.pts[track.track_id][-2]
                            #timestamp += predict_delay(p3, pl,self.shape_img)
                        
            if track.wait_loss and not track.match:
                track.wait_loss = False
                direc = track.direc_
                if draw: frame = self.draw_direc(track.posi_cut, direc, frame)
                pl = self.pts[track.track_id][-2]
                timestamp = frame_id
                timestamp += predict_delay(p3, pl,self.shape_img)
                if timestamp > self.info[-1]: timestamp = self.info[-1]
                sec = np.round(time.time()-self.begin_time, 2)
                data.append([sec,self.info[0], timestamp, direc, self.id])
                #print([sec,self.info[0], timestamp, direc, self.id])
                self.obtracker.tracks.remove(track)
            cv2.putText(frame, str(self.class_names),(int(bbox[0]), int(bbox[1]-5)),0, 5e-3 * 100, (color),2)
        return frame, data
    def draw_direc(self, index, direc, frame):
        r = self.ROI[:-1]
        roi1 = r
        roi2 = r[1:]+r[:1]
        x = (roi1[index][0]+roi2[index][0])//2 + 10
        y = (roi1[index][1]+roi2[index][1])//2 + 10
        cv2.putText(frame, str(direc),(x,y),0, 5e-3 * 300, (0,0,255),3)
        return frame

    def visualize(self,frame):
        self.vis.set_image(frame.copy())
        self.vis.draw_detections(self.detections)
        self.vis.draw_trackers(self.obtracker.tracks)
        return self.vis.return_img()
        
    
    
    
    
    
    
    
        
        
        
