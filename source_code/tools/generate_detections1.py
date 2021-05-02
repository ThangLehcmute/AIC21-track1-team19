import os
import errno
import argparse
import numpy as np
import cv2
from skimage.feature import hog


def HOG_feature(image, boxs):
    x0 = image.shape[1]
    y0 = image.shape[0]
    HOG = []
    for bbox in boxs:
        bbox = np.array(bbox)
        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)
        
        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray([x0, y0]), bbox[2:])
       
        if np.any(bbox[:2] >= bbox[2:]):

            patch = np.random.uniform(0., 255., image.shape).astype(np.uint8)
            HOG.append(patch)
            #return patch
        else:
            sx, sy, ex, ey = bbox
            crop = image[sy:ey, sx:ex]
            crop = cv2.resize(crop,(64,64))
            fd= hog(crop, orientations=9, pixels_per_cell=(16,16),cells_per_block=(3, 3), visualize=False,feature_vector=True, multichannel=True)
            #s = sum(hist)
            #return fd
            HOG.append(fd)
    res = np.array(HOG)
    return res
def create_his(image, boxs): # box [x, y, w, h]
    x0 = image.shape[1]
    y0 = image.shape[0]
    histograms = []
    for bbox in boxs:
        bbox = np.array(bbox)
        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)
        
        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray([x0, y0]), bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
           
            patch = np.random.uniform(0., 255., image.shape).astype(np.uint8)
            histograms.append(patch)
        else:
            sx, sy, ex, ey = bbox
            drop = image[sy:ey, sx:ex]
            drop = cv2.resize(drop,(128,128))
            drop=drop[10:118,10:118]
            hist1 = cv2.calcHist([drop], [0], None, [128], [0, 256])
            hist2 = cv2.calcHist([drop], [1], None, [128], [0, 256])
            hist3 = cv2.calcHist([drop], [2], None, [128], [0, 256])
            hist = (hist1 + hist2 + hist3)/3

            #s = sum(hist)
            hist = np.squeeze(hist)
            hist = hist.T.tolist()
            histograms.append(hist)
    #print(hist)
    #cv2.imshow('ss1',drop)
    res = np.array(histograms)
    return res    
