# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from . import preprocessing
import math


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.95, max_age=20, n_init=1, id_cam=None):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.max_distance_app = 100
        self.max_dis_playoff = 800
        self.kf = kalman_filter.KalmanFilter(id_cam)
        #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        #self.kf_test = KalmanFilter.KalmanFilter(0.28, 1, 1, 1, 11,1)
        self.tracks = []
        self._next_id = 1
            
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)
            #track.du_doan(self.kf_test)

    def update(self, detections, frame_id):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, frame_id)

        # Update track set.
        for track_idx, detection_idx in matches:
            res_update = self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            if not res_update:
              unmatched_detections.append(detection_idx)
              unmatched_tracks.append(track_idx)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed(self.kf, frame_id)
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])


        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections, frame_id):

        def appearance_feature(tracks, detections, track_indices=None, detection_indices=None):
            if track_indices is None:
                track_indices = np.arange(len(tracks))
            if detection_indices is None:
                detection_indices = np.arange(len(detections))
            cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
            for row, track_idx in enumerate(track_indices):
                f_track = tracks[track_idx].loss_frame
                p1 = tracks[track_idx].mean_loss[:2]
                m = tracks[track_idx].mean
                p2 = m[:2]+m[4:6]
                old = tracks[track_idx].hits
                array_dis = []
                for i in detection_indices:
                    p3 = detections[i].to_xyah()[:2]
                    n_frame = 0.8*abs(frame_id - f_track)
                    d2      = 1*np.linalg.norm(p3-p2)
                    if np.linalg.norm(p2-p1) == 0: d = d2
                    else: d = 1.0*np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                    old     = 0.5*1/(old+0.2)     
                    #dis = math.sqrt(d**2+n_frame**2+d2**2+old**2)
                    dis = d+n_frame+d2+old
                    #print('features', d, n_frame, dis)
                    array_dis.append(dis)
                
                cost_matrix[row, :] = np.asarray(array_dis)
            return cost_matrix

        def playoff_feature(tracks, detections, track_indices=None, detection_indices=None):
            if track_indices is None:
                track_indices = np.arange(len(tracks))
            if detection_indices is None:
                detection_indices = np.arange(len(detections))
            cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
            for row, track_idx in enumerate(track_indices):
                m = tracks[track_idx].mean
                p2 = m[:2]
                array_dis = []
                for i in detection_indices:
                    p3 = detections[i].to_xyah()[:2]
                    d2      = 1*np.linalg.norm(p3-p2)
                    #print('features', d2)
                    array_dis.append(d2)
                cost_matrix[row, :] = np.asarray(array_dis)
            return cost_matrix

        iou_track_candidates = [i for i, t in enumerate(self.tracks)]
        unmatched_detections = [i for i,d in enumerate(detections)]

        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        appear_candidates, no_appear_candidates = [], []
        for i in unmatched_tracks_a:
            if ((self.tracks[i].loss_frame != None) and (not self.tracks[i].wait_loss) and  
            (abs(self.tracks[i].mean[4])>2) and (abs(self.tracks[i].mean[5])>2)):
                appear_candidates.append(i)
            else:
                no_appear_candidates.append(i)

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.appearance(appearance_feature,self.max_distance_app,self.tracks,
                detections, appear_candidates, unmatched_detections)

        playoff_unm_tracks = no_appear_candidates + unmatched_tracks_b
        playoff_candidates, all_unmatch_tr = [], []
        for i in playoff_unm_tracks:
            if (self.tracks[i].hits == 1):
                playoff_candidates.append(i)
            else:
                all_unmatch_tr.append(i)  

        matches_c, unmatched_tracks_c, unmatched_detections = \
            linear_assignment.appearance(playoff_feature,self.max_dis_playoff,self.tracks,
                detections, playoff_candidates, unmatched_detections)

        matches = matches_a + matches_b + matches_c
        unmatched_tracks = list(set(unmatched_tracks_c+all_unmatch_tr))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
