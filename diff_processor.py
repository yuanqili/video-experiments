"""
The Processor select frames based on change of low-level property. Factors to
consider for the design of difference processors:

- selection policy
    - dynamic selection
    - first order selection
    - second order selection

- feature extraction
    - image compare: too rough
        - pixel
        - area
        - edge
        - corner

    - image descriptor: too slow...
        - hist
        - hog
        - sift
        - surf

    - block-based compare
        - block pixel
        - block hig
        - block hog
        - block surf
"""

import configparser
import logging
import os
import time
from os.path import exists

import cv2
import imutils
import numpy as np
# import scipy
import scipy.spatial
# from imutils import feature
import skimage.feature as feature

from video_processor import VideoProcessor

CacheValueContainer = {}
diff_logger = logging.getLogger('diff')


class DiffProcessor:
    """
    Args:
        feature (str): type of feature to compare
        selection (str): selection policy includes dynamic, first and second
        thresh (float): threshold value for selection policy, frame with diff
            above which will be send
        fraction (float): only support for first and second, force the fraction
        dataset (str): for loading external config
    """

    def __init__(self, feature='generic', selection='second',
                 thresh=0, fraction=0, dataset=None):
        self.feature_type = feature
        self.fraction = fraction
        self.selection = selection
        self.thresh = thresh
        self.diff_type = f'{feature}-{selection}'.upper()
        section = {}
        if dataset:
            config = configparser.ConfigParser()
            default_config_path = 'config/diff.ini'
            if not exists(default_config_path):
                default_config_path = os.getenv('diff_path')
            if default_config_path is not None:
                config.read(default_config_path)
                if dataset not in config:
                    section = config['default']
                else:
                    section = config[dataset]
        self._load_section(section)

        self.config = {
            'feature': feature,
            'selection': selection,
            'thresh': thresh,
            'fraction': fraction
        }

        if self.fraction and self.selection in ['first', 'second']:
            self.name = f'{self.diff_type}-F{self.fraction:.8f}'
        else:
            self.name = (self.feature_type if self.thresh == 0
                         else f'{self.diff_type}-{self.thresh:.8f}')

    def analyze_video(self, video_path):
        diff_values = []
        with VideoProcessor(video_path) as video:
            prev_frame = next(video)
            prev_frame = self.get_frame_feature(prev_frame)
            for frame in video:
                frame = self.get_frame_feature(frame)
                diff_value = self.cal_frame_diff(frame, prev_frame)
                diff_values.append(diff_value)
                prev_frame = frame
        return diff_values

    def process_video(self, video_path):
        if self.selection == 'dynamic':
            return self.dynamic_selection(video_path)
        elif self.selection == 'first':
            return self.first_order_selection(video_path)
        elif self.selection == 'second':
            return self.second_order_selection(video_path)
        else:
            error_msg = f'Noknown Selection Policy'
            raise KeyError(error_msg)

    def dynamic_selection(self, video_path):
        selected_frames = [1]
        estimations = [1.0]
        time_start = time.time()
        with VideoProcessor(video_path) as video:
            prev_frame = next(video)
            prev_feat = self.get_frame_feature(prev_frame)
            for frame in video:
                feat = self.get_frame_feature(frame)
                dis = self.cal_frame_diff(feat, prev_feat)
                if dis > self.thresh:
                    selected_frames.append(video.index)
                    prev_feat = feat
                    estimations.append(1.0)
                else:
                    estimations.append((self.thresh - dis) / self.thresh)
            total_frames = video.index
        complete_time = time.time() - time_start
        return self._format_selection(selected_frames, total_frames,
                                      complete_time, estimations)

    def first_order_selection(self, video_path):
        global CacheValueContainer
        cache_value_key = f'{self.diff_type}:{str(video_path)}'
        if cache_value_key not in CacheValueContainer:
            diff_logger.info(f'[diff] Create new cache for {cache_value_key}')
            time_start = time.time()
            diff_values = [0.0] + self.analyze_video(video_path)
            diff_time = time.time() - time_start
            CacheValueContainer[cache_value_key] = {
                'diff': diff_values,
                'time': diff_time
            }
        diff_values = CacheValueContainer[cache_value_key]['diff']
        complete_time = CacheValueContainer[cache_value_key]['time']
        total_frames = len(diff_values)

        if self.fraction:
            selected_frames_n = int(self.fraction * total_frames)
            diff_value_argsort = np.argsort(diff_values).tolist()[::-1]
            selected_frames = diff_value_argsort[:selected_frames_n]
            selected_frames = sorted([i + 1 for i in selected_frames])
        else:
            selected_frames = [i + 1 for i in range(len(diff_values))
                               if diff_values[i] > self.thresh]
        if len(selected_frames) == 0:
            selected_frames = [1]
        if selected_frames[0] != 1:
            selected_frames.insert(0, 1)

        if self.fraction:
            max_diff = max(diff_values)
            assert (max_diff != 0)
            estimations = [(1.0 if i + 1 in selected_frames
                            else (max_diff - diff_values[i]) / max_diff)
                           for i in range(total_frames)]
        else:
            estimations = [(1.0 if i + 1 in selected_frames
                            else (self.thresh - diff_values[i]) / self.thresh)
                           for i in range(total_frames)]
        return self._format_selection(selected_frames, total_frames,
                                      complete_time, estimations)

    def second_order_selection(self, video_path):
        global CacheValueContainer
        cache_value_key = f'{self.diff_type}:{str(video_path)}'
        if cache_value_key not in CacheValueContainer:
            diff_logger.info(f'[diff] Create new cache for {cache_value_key}')
            time_start = time.time()
            diff_int = np.cumsum([0.0] + self.analyze_video(video_path)).tolist()
            diff_time = time.time() - time_start
            CacheValueContainer[cache_value_key] = {
                'diff': diff_int,
                'time': diff_time
            }
        diff_int = CacheValueContainer[cache_value_key]['diff']
        complete_time = CacheValueContainer[cache_value_key]['time']
        total_frames = len(diff_int)

        if self.fraction:
            selected_frames_n = np.ceil(self.fraction * total_frames)
            self.thresh = (diff_int[-1] - diff_int[1]) / (selected_frames_n)

        selected_frames = [1]
        estimations = [1.0]
        last, current = 1, 2
        while current < total_frames:
            diff_inc = diff_int[current] - diff_int[last]
            if diff_inc >= self.thresh:
                selected_frames.append(current)
                last = current
                estimations.append(1.0)
            else:
                estimations.append((self.thresh - diff_inc) / self.thresh)
            current += 1
        return self._format_selection(selected_frames, total_frames,
                                      complete_time, estimations)

    def batch_diff(self, diff_value, diff_processors):
        diff_int = np.cumsum([0.0] + diff_value).tolist()
        diff_results = {}
        total_frames = 1 + len(diff_value)
        for dp in diff_processors:
            threshold = dp.thresh
            selected_frames = [1]
            estimations = [1.0]
            last, current = 1, 2
            while current < total_frames:
                diff_inc = diff_int[current] - diff_int[last]
                if diff_inc >= threshold:
                    selected_frames.append(current)
                    last = current
                    estimations.append(1.0)
                else:
                    estimations.append((threshold - diff_inc) / threshold)
                current += 1
            diff_results[dp.name] = self._format_selection(selected_frames, total_frames, 0, estimations)
        return diff_results

    def cal_frame_diff(self, frame, prev_frame):
        """ Calculate the different between frames """
        raise NotImplementedError()

    def get_frame_feature(self, frame):
        """ Extract feature of frame """
        raise NotImplementedError()

    def get_config(self, key='thresh'):
        """ Get the configuration, e.g., parameter or threshold """
        return self.config[key]

    def get_feature_type(self):
        return self.feature_type

    def _format_selection(self, selected_frames, total_frames,
                          complete_time, estimations):
        return {
            'fps': total_frames / complete_time if complete_time != 0 else -1,
            'selected_frames': selected_frames,
            'num_selected_frames': len(selected_frames),
            'fraction': len(selected_frames) / total_frames,
            'estimation': sum(estimations) / len(estimations)
        }

    def _load_section(self, section):
        return

    def __str__(self):
        return self.name


class PixelDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__('pixel', selection, thresh, fraction, dataset)

    def get_frame_feature(self, frame):
        return frame

    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_diff = cv2.absdiff(frame, prev_frame)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.threshold(frame_diff, self.pixel_thresh_low_bound,
                                   255, cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.pixel_thresh_low_bound = (21 if 'PIXEL_THRESH_LOW_BOUND'
                                             not in section else section.getint('PIXEL_THRESH_LOW_BOUND'))


class AreaDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__(feature='area', selection=selection, thresh=thresh,
                         fraction=fraction, dataset=dataset)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,
                                (self.area_blur_rad, self.area_blur_rad), self.area_blur_var)
        return blur

    def cal_frame_diff(self, frame, prev_frame):
        total_pixels = frame.shape[0] * frame.shape[1]
        frame_delta = cv2.absdiff(frame, prev_frame)
        thresh = cv2.threshold(frame_delta, self.area_thresh_low_bound,
                               255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None)
        contours = cv2.findContours(thresh.copy(),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if not contours:
            return 0.0
        return max([cv2.contourArea(c) / total_pixels for c in contours])

    def _load_section(self, section):
        self.area_blur_rad = (11 if 'AREA_BLUR_RAD'
                                    not in section else section.getint('AREA_BLUR_RAD'))
        self.area_blur_var = (0 if 'AREA_BLUR_VAR'
                                   not in section else section.getint('EDGE_BLUR_VAR'))
        self.area_thresh_low_bound = (21 if 'AREA_THRESH_LOW_BOUND'
                                            not in section else section.getint('AREA_THRESH_LOW_BOUND'))


class EdgeDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__('edge', selection, thresh, fraction, dataset)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.edge_blur_rad, self.edge_blur_rad),
                                self.edge_blur_var)
        edge = cv2.Canny(blur, self.edge_canny_low, self.edge_canny_high)
        return edge

    def cal_frame_diff(self, edge, prev_edge):
        total_pixels = edge.shape[0] * edge.shape[1]
        frame_diff = cv2.absdiff(edge, prev_edge)
        frame_diff = cv2.threshold(frame_diff, self.edge_thresh_low_bound,
                                   255, cv2.THRESH_BINARY)[1]
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.edge_blur_rad = (11 if 'EDGE_BLUR_RAD'
                                    not in section else section.getint('EDGE_BLUR_RAD'))
        self.edge_blur_var = (0 if 'EDGE_BLUR_VAR'
                                   not in section else section.getint('EDGE_BLUR_VAR'))
        self.edge_canny_low = (101 if 'EDGE_CANNY_LOW'
                                      not in section else section.getint('EDGE_CANNY_LOW'))
        self.edge_canny_high = (255 if 'EDGE_CANNY_HIGH'
                                       not in section else section.getint('EDGE_CANNY_HIGH'))
        self.edge_thresh_low_bound = (21 if 'EDGE_THRESH_LOW_BOUND'
                                            not in section else section.getint('EDGE_THRESH_LOW_BOUND'))


class CornerDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__('corner', selection, thresh, fraction, dataset)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner = cv2.cornerHarris(gray, self.corner_block_size, self.corner_ksize, self.corner_k)
        corner = cv2.dilate(corner, None)
        return corner

    def cal_frame_diff(self, corner, prev_corner):
        total_pixels = corner.shape[0] * corner.shape[1]
        frame_diff = cv2.absdiff(corner, prev_corner)
        changed_pixels = cv2.countNonZero(frame_diff)
        fraction_changed = changed_pixels / total_pixels
        return fraction_changed

    def _load_section(self, section):
        self.corner_block_size = (5 if 'CORNER_BLOCK_SIZE'
                                       not in section else section.getint('CORNER_BLOCK_SIZE'))
        self.corner_ksize = (3 if 'CORNER_KSIZE'
                                  not in section else section.getint('CORNER_KSIZE'))
        self.corner_k = (0.05 if 'CORNER_K'
                                 not in section else section.getfloat('CORNER_K'))


class HistDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__('hist', selection, thresh, fraction, dataset)

    def get_frame_feature(self, frame):
        nb_channels = frame.shape[-1]
        hist = np.zeros((self.hist_nb_bins * nb_channels, 1), dtype='float32')
        for i in range(nb_channels):
            hist[i * self.hist_nb_bins: (i + 1) * self.hist_nb_bins] = \
                cv2.calcHist(frame, [i], None, [self.hist_nb_bins], [0, 256])
        hist = cv2.normalize(hist, hist)
        return hist

    def cal_frame_diff(self, frame, prev_frame):
        return cv2.compareHist(frame, prev_frame, cv2.HISTCMP_CHISQR)

    def _load_section(self, section):
        self.hist_nb_bins = (32 if 'HIST_NB_BINS'
                                   not in section else section.getint('HIST_NB_BINS'))


class HOGDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__(feature='hog', selection=selection, thresh=thresh,
                         fraction=fraction, dataset=dataset)

    def get_frame_feature(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to speed up
        gray = cv2.resize(gray, (self.hog_resize, self.hog_resize))
        hog = feature.hog(
            gray, orientations=self.hog_orientations,
            pixels_per_cell=(self.hog_pixel_cell, self.hog_pixel_cell),
            cells_per_block=(self.hog_cell_block, self.hog_cell_block)).astype('float32')
        return hog

    def cal_frame_diff(self, frame, prev_frame):
        dis = scipy.spatial.distance.euclidean(frame, prev_frame)
        dis /= frame.shape[0]
        return dis

    def _load_section(self, section):
        self.hog_resize = (512 if 'HOG_RESIZE'
                                  not in section else section.getint('HOG_RESIZE'))
        self.hog_orientations = (10 if 'HOG_ORIENTATIONS'
                                       not in section else section.getint('HOG_ORIENTATIONS'))
        self.hog_pixel_cell = (5 if 'HOG_PIXEL_CELL'
                                    not in section else section.getint('HOG_PIXEL_CELL'))
        self.hog_cell_block = (2 if 'HOG_CELL_BLOCK'
                                    not in section else section.getint('HOG_CELL_BLOCK'))


class SIFTDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__(feature='sift', selection=selection, thresh=thresh,
                         fraction=fraction, dataset=dataset)

    def get_frame_feature(self, frame):
        sift = cv2.xfeatures2d.SIFT_create()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(gray, None)
        des = np.mean(des, axis=0).astype('float32') if des is not None else np.zeros(128)
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = scipy.spatial.distance.euclidean(frame, prev_frame)
        dis /= frame.shape[0]
        return dis


class SURFDiff(DiffProcessor):

    def __init__(self, selection='dynamic', thresh=0, fraction=0, dataset=None):
        super().__init__(feature='surf', selection=selection, thresh=thresh,
                         fraction=fraction, dataset=dataset)

    def get_frame_feature(self, frame):
        surf = cv2.xfeatures2d.SURF_create()
        surf.setUpright(True)
        surf.setHessianThreshold(self.surf_hessian_thresh)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, des = surf.detectAndCompute(gray, None)
        des = np.mean(des, axis=0).astype('float32') if des is not None else np.zeros(128)
        return des

    def cal_frame_diff(self, frame, prev_frame):
        dis = scipy.spatial.distance.euclidean(frame, prev_frame)
        dis /= frame.shape[0]
        return dis

    def _load_section(self, section):
        self.surf_hessian_thresh = (400 if 'SURF_HESSIAN_THRESH' not in section
                                    else section.getint('SURF_HESSIAN_THRESH'))


def build_diff_processor(diff_type, selection='second',
                         thresh=0, fraction=0, dataset=None):
    if diff_type == 'pixel':
        return PixelDiff(selection, thresh, fraction, dataset)
    elif diff_type == 'area':
        return AreaDiff(selection, thresh, fraction, dataset)
    elif diff_type == 'edge':
        return EdgeDiff(selection, thresh, fraction, dataset)
    elif diff_type == 'corner':
        return CornerDiff(selection, thresh, fraction, dataset)
    else:
        error_msg = f'Unknown diff type {diff_type}'
        raise KeyError(error_msg)
