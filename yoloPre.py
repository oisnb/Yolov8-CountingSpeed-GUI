from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.cfg import get_cfg
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, QThread, Signal, QObject, QPoint, Qt
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from lprr import CHARS
from ui.CustomMessageBox import MessageBox
from ui.home import Ui_MainWindow
from UIFunctions import *
from collections import defaultdict, deque

from pathlib import Path
from utils.capnums import Camera
from utils.rtsp_win import Window
from PIL import Image
from lprr.plate import de_lpr,dr_plate
from numpy import random
import numpy as np
import math
import time
import json
import torch
import sys
import cv2
import os

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}

object_counter1 = {}

line = [(100, 500), (1050, 500)]
speed_line_queue = {}


def estimatespeed(Location1, Location2):
    # Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel / ppm
    time_constant = 15 * 3.6
    # distance = speed/time
    speed = d_meters * time_constant

    return int(speed)


def init_tracker():
    global deepsort, object_counter, object_counter1
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    object_counter = {}

    object_counter1 = {}

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    cv2.line(img, (50, 500), (1200, 500), (46, 162, 112), 6)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # add center to buffer 这里加入速度
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1

        try:
            label = label + " " + str(sum(speed_line_queue[id]) // len(speed_line_queue[id])) + "km/h"
        except:
            pass
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

        # 4. Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" + str(value)
            cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.line(img, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)

        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" + str(value)
            cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return img

class YoloPredictor(BasePredictor, QObject):
    yolo2main_pre_img = Signal(np.ndarray)   # raw image signal
    yolo2main_res_img = Signal(np.ndarray)   # test result signal
    yolo2main_status_msg = Signal(str)       # Detecting/pausing/stopping/testing complete/error reporting signal
    yolo2main_fps = Signal(str)              # fps
    yolo2main_labels = Signal(dict)          # Detected target results (number of each category)
    yolo2main_progress = Signal(int)         # Completeness
    yolo2main_class_num = Signal(int)        # Number of categories detected
    yolo2main_target_num = Signal(int)       # Targets detected

    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super(YoloPredictor, self).__init__()
        QObject.__init__(self)

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # GUI args
        self.used_model_name = None      # The detection model name to use
        self.new_model_name = None       # Models that change in real time
        self.source = ''                 # input source
        self.stop_dtc = False            # Termination detection
        self.continue_dtc = True         # pause
        self.save_res = False            # Save test results
        self.save_txt = False            # save label(txt) file
        self.iou_thres = 0.45            # iou
        self.conf_thres = 0.25           # conf
        self.speed_thres = 10            # delay, ms
        self.labels_dict = {}            # return a dictionary of results
        self.progress_value = 0          # progress bar


        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks
        callbacks.add_integration_callbacks(self)
        self.ch_image = 0 #选图片或视频
        self.ch_video = 0


    # main for detect
    @smart_inference_mode()
    def video_run(self):
        try:
            init_tracker()
            if self.args.verbose:
                LOGGER.info('')

            # set model
            self.yolo2main_status_msg.emit('Loding Model...')
            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            # set source
            self.setup_source(self.source if self.source is not None else self.args.source)

            # Check save path/label
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # warmup model
            if not self.done_warmup:
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None

            # start detection
            # for batch in self.dataset:


            count = 0                       # run location frame
            start_time = time.time()        # used to calculate the frame rate
            batch = iter(self.dataset)
            while True:
                # Termination detection
                self.ch_video = 1
                if self.stop_dtc:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection terminated!')
                    break

                # Change the model midway
                if self.used_model_name != self.new_model_name:
                    # self.yolo2main_status_msg.emit('Change Model...')
                    self.setup_model(self.new_model_name)
                    self.used_model_name = self.new_model_name

                # pause switch
                if self.continue_dtc:
                    # time.sleep(0.001)
                    self.yolo2main_status_msg.emit('Detecting...')
                    batch = next(self.dataset)  # next data

                    self.batch = batch
                    path, im, im0s, vid_cap, s = batch
                    visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

                    # Calculation completion and frame rate (to be optimized)
                    count += 1              # frame count +1
                    if vid_cap:
                        all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)   # total frames
                    else:
                        all_count = 1
                    self.progress_value = int(count/all_count*1000)         # progress bar(0~1000)
                    if count % 5 == 0 and count >= 5:                     # Calculate the frame rate every 5 frames
                        self.yolo2main_fps.emit(str(int(5/(time.time()-start_time))))
                        start_time = time.time()

                    # preprocess
                    with self.dt[0]:
                        im = self.preprocess(im)
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim
                    # inference
                    with self.dt[1]:
                        preds = self.model(im, augment=self.args.augment, visualize=visualize)
                    # postprocess
                    with self.dt[2]:
                        self.results = self.postprocess(preds, im, im0s)

                    # visualize, save, write results
                    n = len(im)     # To be improved: support multiple img
                    for i in range(n):
                        self.results[i].speed = {
                            'preprocess': self.dt[0].dt * 1E3 / n,
                            'inference': self.dt[1].dt * 1E3 / n,
                            'postprocess': self.dt[2].dt * 1E3 / n}
                        p, im0 = (path[i], im0s[i].copy()) if self.source_type.from_img \
                            else (path, im0s.copy())
                        p = Path(p)     # the source dir

                        # s:::   video 1/1 (6/6557) 'path':
                        # must, to get boxs\labels
                        label_str, img = self.write_results(i, self.results, (p, im, im0))   # labels   /// original :s +=

                        # labels and nums dict
                        class_nums = 0
                        target_nums = 0
                        self.labels_dict = {}

                       # print("label_str: ", label_str)
                        if 'no detections' in label_str:
                            pass
                        else:
                            for ii in label_str.split(',')[:-1]:
                                nums, label_name = ii.split('~')
                                self.labels_dict[label_name] = int(nums)
                                target_nums += int(nums)
                                class_nums += 1

                        # save img or video result
                        if self.save_res:
                            self.save_preds(vid_cap, i, str(self.save_dir / p.name))

                        # Send test results
                        #print("img!!!:", img)
                        if type(img) == int:
                            self.yolo2main_res_img.emit(im0) # after detection
                        else:
                            self.yolo2main_res_img.emit(img)
                        self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0]) # Before testing
                        # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
                        self.yolo2main_class_num.emit(class_nums)
                        self.yolo2main_target_num.emit(target_nums)

                        if self.speed_thres != 0:
                            time.sleep(self.speed_thres/1000)   # delay , ms

                    self.yolo2main_progress.emit(self.progress_value)   # progress bar

                # Detection completed
                if count + 1 >= all_count:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection completed')
                    break

        except Exception as e:
            pass
            print(e)
            self.yolo2main_status_msg.emit('%s' % e)


    @smart_inference_mode()
    def image_run(self):
        try:
            if self.args.verbose:
                LOGGER.info('')

            # set model
            self.yolo2main_status_msg.emit('Loding Model...')

            if not self.model:
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            # set source  [视频资源]
            self.setup_source(self.source if self.source is not None else self.args.source)

            # Check save path/label
            if self.save_res or self.save_txt:
                (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # warmup model
            # 热身模型
            if not self.done_warmup:
                # 调用模型的 warmup 函数，其中 imgsz 参数为输入图像的大小
                # 如果模型使用 PyTorch，imgsz 参数应为 [batch_size, channels, height, width]
                # 如果模型使用 Triton，imgsz 参数应为 [height, width, channels, batch_size]
                self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
                # 将 done_warmup 标记为 True，以标记模型已经热身过
                self.done_warmup = True

            self.seen, self.windows, self.dt, self.batch = 0, [], (ops.Profile(), ops.Profile(), ops.Profile()), None
            # 创建名为 dt 的实例变量，用于存储一个元组，并将其初始化为包含三个对象 ops.Profile() 的元组。
            # ops.Profile() 是指从 ops 模块中导入名为 Profile() 的对象。

            print('start detection')
            # start detection
            # for batch in self.dataset:

            count = 0  # run location frame
            start_time = time.time()  # used to calculate the frame rate
            batch = iter(self.dataset)

            self.ch_image = 1

            # Change the model midway 【切换model】  如果不相等，则执行 setup_model() 方法设置新的模型
            if self.used_model_name != self.new_model_name:
                # self.yolo2main_status_msg.emit('Change Model...')
                self.setup_model(self.new_model_name)
                self.used_model_name = self.new_model_name

            # pause switch  用于控制程序的暂停和继续
            if self.continue_dtc:
                # time.sleep(0.001)
                self.yolo2main_status_msg.emit('Detecting...')
                batch = next(self.dataset)  # next data

                self.batch = batch
                path, im, im0s, vid_cap, s = batch
                visualize = increment_path(self.save_dir / Path(path).stem,
                                           mkdir=True) if self.args.visualize else False

                # Calculation completion and frame rate (to be optimized)
                count += 1  # frame count +1
                if vid_cap:
                    all_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)  # total frames
                else:
                    all_count = 1
                self.progress_value = int(count / all_count * 1000)  # progress bar(0~1000)
                if count % 5 == 0 and count >= 5:  # Calculate the frame rate every 5 frames
                    self.yolo2main_fps.emit(str(int(5 / (time.time() - start_time))))
                    start_time = time.time()

                # preprocess
                # self.dt 包含了三个 DetectorTime 类型的对象，表示预处理、推理和后处理所花费的时间

                ## 使用 with 语句记录下下一行代码所花费的时间，self.dt[0] 表示记录预处理操作所花费的时间。
                with self.dt[0]:
                    # 调用 self.preprocess 方法对图像进行处理，并将处理后的图像赋值给 im 变量。
                    im = self.preprocess(im)
                    # 如果 im 的维度为 3（RGB 图像），则表示这是一张单张图像，需要将其扩展成 4 维，加上 batch 维度。
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim  扩大批量调暗
                # inference
                with self.dt[1]:
                    # 调用模型对图像进行推理，并将结果赋值给 preds 变量。
                    preds = self.model(im, augment=self.args.augment, visualize=visualize)
                # postprocess
                with self.dt[2]:
                    # 调用 self.postprocess 方法对推理结果进行后处理，并将结果保存到 self.results 变量中。
                    # 其中 preds 是模型的预测结果，im 是模型输入的图像，而 im0s 是原始图像的大小。
                    self.results = self.postprocess(preds, im, im0s)

                # visualize, save, write results
                n = len(im)  # To be improved: support multiple img
                for i in range(n):
                    self.results[i].speed = {
                        'preprocess': self.dt[0].dt * 1E3 / n,
                        'inference': self.dt[1].dt * 1E3 / n,
                        'postprocess': self.dt[2].dt * 1E3 / n}
                    p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                        else (path, im0s.copy())

                    p = Path(p)  # the source dir

                    # s:::   video 1/1 (6/6557) 'path':
                    # must, to get boxs\labels
                    label_str = self.write_results(i, self.results, (p, im, im0))  # labels   /// original :s +=
                    #print(label_str) #个数~物体

                    # labels and nums dict
                    class_nums = 0
                    target_nums = 0
                    self.labels_dict = {}
                    if 'no detections' in label_str:
                        pass
                    else:
                        for ii in label_str.split(',')[:-1]:
                            nums, label_name = ii.split('~')
                            self.labels_dict[label_name] = int(nums)
                            target_nums += int(nums)
                            class_nums += 1

                    # save img or video result
                    if self.save_res:
                        self.save_preds(vid_cap, i, str(self.save_dir / p.name))


                    self.yolo2main_res_img.emit(im0)  # after detection  ----------结果
                    self.yolo2main_pre_img.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])  # Before testing
                    # self.yolo2main_labels.emit(self.labels_dict)        # webcam need to change the def write_results
                    self.yolo2main_class_num.emit(class_nums)
                    self.yolo2main_target_num.emit(target_nums)

                    print('send success!')

                    # if self.speed_thres != 0:
                    #     time.sleep(self.speed_thres / 1)  # delay , ms

                self.yolo2main_progress.emit(self.progress_value)  # progress bar

                # Detection completed
                if count + 1 >= all_count:
                    if isinstance(self.vid_writer[-1], cv2.VideoWriter):
                        self.vid_writer[-1].release()  # release final video writer
                    self.yolo2main_status_msg.emit('Detection completed')


        except Exception as e:
            print(e)
            self.yolo2main_status_msg.emit('%s' % e)


    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        ### important
        preds = ops.non_max_suppression(preds,
                                        self.conf_thres,
                                        self.iou_thres,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        if self.ch_image == 1:
            # idx 是一个整数，用于指定批处理中的某个图像；
            # results 是一个列表，其中包含在模型中对 batch 执行前向传递的结果；
            # batch 是模型输入的一个批处理张量，它由三个元素组成：p，im 和 im0。

            # 将元组 batch 中三个变量解包存储到 p、im 和 im0 中。
            p, im, im0 = batch
            log_string = ''

            # 用于判断输入图像的形状（shape）是否为 3D，如果是则在前面添加一个新的维度，以表示批处理。这是一种处理不同大小的输入图像的常见方法。
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            self.seen += 1

            # 使用 if/else 语句设置变量 imc，该变量存储输入图像的一个副本。
            # 如果参数 save_crop（在 self.args 中）为真，则存储裁剪的图像，否则存储原始图像。
            imc = im0.copy() if self.args.save_crop else im0

            # 检查输入数据是否来自网络摄像头或图像文件
            if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1         # attention
                log_string += f'{idx}: '
                frame = self.dataset.count
            else:
                frame = getattr(self.dataset, 'frame', 0)
            self.data_path = p
            self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
            # log_string += '%gx%g ' % im.shape[2:]         # !!! don't add img size~
            self.annotator = self.get_annotator(im0)

            # 统计检测到的目标数量和种类的段落
            det = results[idx].boxes  # TODO: make boxes inherit from tensors

            if len(det) == 0:
                return f'{log_string}(no detections), '  # if no, send this~~

            # 添加当前目标数量和名称到日志字符串
            # 【det.cls.unique() 方法返回了 det.cls 列中的所有唯一值】
            for c in det.cls.unique():
                # det.cls == c 这个条件判断表达式会返回一个由布尔值组成的数组
                n = (det.cls == c).sum()  # detections per class

                # it only recognizes license-plates and records the total number of license-plates
                if(self.model.names[int(c)] == 'license-plate'):
                    log_string = f"{n}~{self.model.names[int(c)]},"  # {'s' * (n > 1)}, "   # don't add 's'

            # now log_string is the classes 👆
            # print(log_string)

            # write & save & draw
            for d in reversed(det):
                cls, conf = d.cls.squeeze(), d.conf.squeeze()

                # 获取类别  get category
                c = int(cls)  # integer class
                name = f'id:{int(d.id.item())} {self.model.names[c]}' if d.id is not None else self.model.names[c]

                # 如果不是车牌，则跳过！
                # if there is not a license-plate, jump it
                if (name != 'license-plate'):
                    continue

                # 画车牌 draw a license plate 从这里开始把车牌单独标出来
                plate = de_lpr(d.xyxy.squeeze(), im0)
                plate = np.array(plate)
                car_number_laber = ""
                for i in range(0, plate.shape[1]):
                    b = CHARS[plate[0][i]]
                    car_number_laber += b
                print("plate:  ", car_number_laber)

                if self.save_txt:  # Write to file 写入文本文件

                    line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                        if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                    with open(f'{self.txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # 检测结果绘制到图像上，并显示出来。
                if self.save_res or self.args.save_crop or self.args.show or True:  # Add bbox to image(must)

                    # 如果 self.args.hide_labels = True，则为 None
                    # 否则 (name if self.args.hide_conf else f'{name} {conf:.2f}')
                    # 如果 self.args.hide_conf = True，则为 name
                    # 否则 f'{name} {conf:.2f}'
                    self.annotator.box_label(d.xyxy.squeeze(), car_number_laber, color=colors(c, True)) #画框框与标注

                    # 原标签 original label
                    # label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')


                # 将画在图像上的边界框区域保存为一个单独的图像或者视频文件
                if self.args.save_crop:
                    save_one_box(d.xyxy,
                                 imc,
                                 file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                                 BGR=True)
            self.ch_image = 0 #置0
            return log_string

        elif self.ch_video == 1:
            img = 0
            p, im, im0 = batch
            all_outputs = []
            log_string = ""
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            self.seen += 1
            im0 = im0.copy()
            # if self.webcam:  # batch_size >= 1
            #     log_string += f'{idx}: '
            #     frame = self.dataset.count
            # else:
            frame = getattr(self.dataset, 'frame', 0)

            self.data_path = p
            save_path = str(self.save_dir / p.name)  # im.jpg
            self.txt_path = str(self.save_dir / 'labels' / p.stem) + (
                '' if self.dataset.mode == 'image' else f'_{frame}')
            #log_string += '%gx%g ' % im.shape[2:]  # print string
            self.annotator = self.get_annotator(im0)

            det = results[idx].boxes
            all_outputs.append(det)
            if len(det) == 0:
                return log_string, img
            for c in det.cls.unique():
                n = (det.cls == c).sum()  # detections per class
                log_string+= f"{n}~{self.model.names[int(c)]}," #{'s' * (n > 1)}

            # write
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            xywh_bboxs = []
            confs = []
            oids = []
            outputs = []

            for d in reversed(det):
                xyxy = d.xyxy.squeeze()
                conf = d.conf.squeeze()
                cls = d.cls.squeeze()
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
                oids.append(int(cls))
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)

            outputs = deepsort.update(xywhs, confss, oids, im0)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]

                img = draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)

            return log_string, img


