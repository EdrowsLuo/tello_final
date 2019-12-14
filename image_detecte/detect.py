#!/usr/bin/env python
import os
import random
import shutil
import time
from sys import platform

import cv2
import numpy as np
import torch

import sl4p
from image_detecte.models import ONNX_EXPORT, Darknet, load_darknet_weights, load_classes, parse_data_cfg, \
    scale_coords, plot_one_box, non_max_suppression, scale_coords, plot_one_box
import threading
import locks
from utils import torch_utils


class DetectResult:
    def __init__(self, x1, y1, x2, y2, object_conf, class_conf, class_idx, class_name, color):
        self.x1, self.y1, self.x2, self.y2, self.object_conf, self.class_conf, self.class_idx, self.class_name, self.color = \
            x1, y1, x2, y2, object_conf, class_conf, class_idx, class_name, color

    def __str__(self):
        return "[(x1, y1, x1, y2) = (%d, %d, %d, %d), conf = (%.2f, %.2f), %s\ncolor = %s]" \
               %(
               self.x1, self.y1, self.x2, self.y2, self.object_conf, self.class_conf, self.class_name, str(self.color))


class ResultCollection:
    def __init__(self):
        self.map = {}
        self.call_times = 0

    def add_result(self, result: DetectResult):
        if result.object_conf < 0.25:
            return
        if result.class_name not in self.map:
            self.map[result.class_name] = []
        self.map[result.class_name].append(result)

    def add_all_results(self, results):
        self.call_times += 1
        if results is None:
            return
        for rr in results:
            self.add_result(rr)

    def get_result_collection(self):
        result_collect = {}
        for ss in self.map:
            count = 0
            conf = 0
            obj_conf = 0
            max_conf = 0
            for rs in self.map[ss]:
                count += 1
                conf += rs.class_conf
                obj_conf += rs.object_conf
                if rs.object_conf > max_conf:
                    max_conf = rs.object_conf
            conf = conf/count
            obj_conf = obj_conf/count
            result_collect[ss] = {
                'name': ss,
                'call_times': self.call_times,
                'count': count,
                'max_conf': float(max_conf),
                'object_conf': float(obj_conf),
                'class_conf': float(conf)
            }
        return result_collect


class Detect:
    def __init__(self, conf, weights='fire_new_1207.pt'):
        # Initialize this once
        main_dir = os.path.split(os.path.abspath(__file__))[0]
        self.main_dir = main_dir
        cfg = os.path.join(main_dir, 'cfg/yolov3.cfg')
        weights = os.path.join(main_dir, 'weights/%s' % weights)
        output = os.path.join(main_dir, 'data/output')
        img_size = 416
        self.conf_thres = conf

        device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
        torch.backends.cudnn.benchmark = False  # set False for reproducible results
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        # Initialize model
        if ONNX_EXPORT:
            s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
            model = Darknet(cfg, s)
        else:
            model = Darknet(cfg, img_size)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3, s[0], s[1]))
            torch.onnx.export(model, img, os.path.join(main_dir, 'weights/export.onnx'), verbose=True)
            return
        self.model = model
        self.device = device
        # Eval mode
        self.model.to(device).eval()
        self.colors = None

    def detect(self, img):
        # type: (object) -> list[DetectResult]
        # Initialized  for every detection
        time.sleep(0.1)
        data = os.path.join(self.main_dir, 'data/fire.data')
        output = os.path.join(self.main_dir, 'data/output')
        img_size = 416
        nms_thres = 0.5
        save_txt = False
        save_images = False
        save_path = os.path.join(self.main_dir, 'data/output/result.jpg')
        # Set Dataloader
        img0 = img  # BGR

        # Padded resize
        tmpresultimg = self.letterbox(img0, new_shape=img_size)
        img = tmpresultimg[0]
        # cv2.imshow("",img)
        # cv2.waitKey(0)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Get classes and colors
        classes = load_classes(os.path.join(self.main_dir, parse_data_cfg(data)['names']))
        if self.colors is None:
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
        colors = self.colors

        # Run inference
        # t0 = time.time()

        # Get detections

        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        # print("img.shape")
        # print(img.shape )
        pred, _ = self.model(img)
        det = non_max_suppression(pred.float(), self.conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        if det is None:
            return []
        if det.shape[0] <= 0:
            return []
        else:
            result = []
            for x in range(len(det)):
                result.append(DetectResult(det[x][0], det[x][1], det[x][2], det[x][3],
                                           det[x][4], det[x][5], int(det[x][6]), classes[int(det[x][6])],
                                           colors[int(det[x][6])]))
            return result

    def draw_result(self, img, det, show=False):
        if det is None or len(det) == 0:
            if show:
                locks.imshow('result', img)
            return
        for det_pack in det:
            xyxy = []
            for c in [det_pack.x1, det_pack.y1, det_pack.x2, det_pack.y2]:
                xyxy.append(c)
            conf = det_pack.class_conf
            label = '%s %.2f %.2f'%(det_pack.class_name, conf, det_pack.object_conf)
            plot_one_box(xyxy, img, label=label, color=det_pack.color)
        if show:
            locks.imshow('result', img)

    def letterbox(self, img, new_shape=416, color=(128, 128, 128), mode='auto'):
        # Resize a rectangular image to a 32 pixel multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            ratio = float(new_shape)/max(shape)
        else:
            ratio = max(new_shape)/max(shape)  # ratio  = new / old
        ratiow, ratioh = ratio, ratio
        new_unpad = (int(round(shape[1]*ratio)), int(round(shape[0]*ratio)))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if mode is 'auto':  # minimum rectangle
            dw = np.mod(new_shape - new_unpad[0], 32)/2  # width padding
            dh = np.mod(new_shape - new_unpad[1], 32)/2  # height padding
        elif mode is 'square':  # square
            dw = (new_shape - new_unpad[0])/2  # width padding
            dh = (new_shape - new_unpad[1])/2  # height padding
        elif mode is 'rect':  # square
            dw = (new_shape[1] - new_unpad[0])/2  # width padding
            dh = (new_shape[0] - new_unpad[1])/2  # height padding
        elif mode is 'scaleFill':
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape, new_shape)
            ratiow, ratioh = new_shape/shape[1], new_shape/shape[0]

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad,
                             interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return (img, ratiow, ratioh, dw, dh)


class Task:
    def __init__(self, task):
        self.task = task
        self.finish = False
        self.result = None

    def run(self):
        self.task()
        self.finish = True

    def invoke_on(self, looper):
        looper.add_task(self)
        while not self.finish:
            time.sleep(0.01)
        return self.result


class DetectImage(Task):

    def __init__(self, detector, img):
        Task.__init__(self, task=self.detect_img)
        self.img = img
        self.detector = detector

    def detect_img(self, idx=0):
        if idx > 10:
            raise BaseException()
        try:
            self.result = self.detector.detect(self.img)
        except RuntimeError:
            time.sleep(0.3)
            self.detect_img(idx=idx + 1)


class TaskLoop:

    def __init__(self, flag=None):
        self.lock = threading.Lock()
        self.tasks = []
        if flag is None:
            def f():
                return True
            flag = f
        self.flag = flag

    def add_task(self, task):
        self.lock.acquire()
        try:
            self.tasks.append(task)
        finally:
            self.lock.release()

    def start(self):
        while self.flag():
            self.lock.acquire()
            try:
                while len(self.tasks) > 0:
                    self.tasks.pop(0).run()
            finally:
                self.lock.release()
            time.sleep(0.01)
        for t in self.tasks:
            t.run()

    def start_async(self, daemon=True):
        t = threading.Thread(target=self.start)
        t.daemon = daemon
        t.start()


def async_main():
    loop = TaskLoop()
    detector = Detect(0.1)

    def mm():
        logger = sl4p.Sl4p("__main__", "1")
        logger.info("start")
        logger.info("CUDA %s"%str(torch.cuda.is_available()))
        test_files = [0, 1, 2, 3, 4]
        for s in test_files:
            img = cv2.imread(os.path.join(detector.main_dir, 'data/samples/%s.jpg'%s))
            start = time.time()
            logger.info("start detect %s"%s)
            result_obj = DetectImage(detector, img).invoke_on(loop)
            # detector.draw_result(img, result_obj)
            end = time.time()
            logger.info("time: " + str(end - start) + "s")
            for r in result_obj:
                logger.info(str(r))

    t = threading.Thread(target=mm)
    t.daemon = True
    t.start()
    loop.start()


if __name__ == '__main__':
    async_main()
