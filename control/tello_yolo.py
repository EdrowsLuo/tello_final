# coding=utf-8
import cv2
import sl4p
from control import tello_center, tello_abs, tello_image_process
from image_detecte.detect import Detect
import threading
import time
import numpy as np


class YoloService(tello_center.Service):
    name = 'yolo_service'
    PRELOAD_YOLO_DETECTOR = 'yolo_detector'
    CONFIG_LOOP_DETECTION = 'yolo::loop_detection'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.logger = sl4p.Sl4p(YoloService.name)
        self.backend = tello_center.service_proxy_by_class(tello_abs.TelloBackendService)  # type: tello_abs.TelloBackendService

    @staticmethod
    def preload(service: tello_center.PreLoadService):
        service.put_loaded(YoloService.PRELOAD_YOLO_DETECTOR, Detect(0.1))

    def detect(self, img, x=0, y=0, w=0, h=0):
        detector = tello_center.get_preloaded(YoloService.PRELOAD_YOLO_DETECTOR)  # type: Detect
        if detector is None:
            raise BaseException('detector not loaded')
        if w == 0 or h == 0:
            return detector.detect(img)
        else:
            c = np.copy(img[y:y+h, x:x+w])
            cv2.imshow('copy', c)
            cv2.waitKey(1)
            results = detector.detect(c)
            if results is None:
                return None
            for r in results:
                r.x1 += x
                r.x2 += x
                r.y1 += y
                r.y2 += y
            return results

    def get_detector(self) -> Detect:
        return tello_center.get_preloaded(YoloService.PRELOAD_YOLO_DETECTOR)  # type: Detect

    def loop_thread(self):
        while self.flag():
            detector = tello_center.get_preloaded(YoloService.PRELOAD_YOLO_DETECTOR)  # type: Detect
            if self.backend.available() and detector is not None:
                img, _ = self.backend.drone.get_image_and_state()
                if img is None:
                    time.sleep(0.1)
                    continue
                img = np.copy(img)
                result = detector.detect(img)
                #result = self.detect(img, 480, 100, 480, 360)
                detector.draw_result(img, result)
                cv2.imshow('detect', img)
                cv2.waitKey(1)
                time.sleep(0.3)
            else:
                time.sleep(0.1)
        self.logger.info('exit loop detection')

    def start(self):
        if tello_center.get_config(YoloService.CONFIG_LOOP_DETECTION, fallback=True):
            self.logger.info('start loop detection')
            t = threading.Thread(target=self.loop_thread)
            t.daemon = True
            t.start()


if __name__ == '__main__':
    tello_center.register_service(tello_center.ConfigService(config={
        tello_abs.TelloBackendService.CONFIG_STOP: True,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: True,

        YoloService.CONFIG_LOOP_DETECTION: True,

        # FPS config
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_VIDEO_FPS): False,
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_STATE_FPS): False,
        tello_center.FpsRecoder.key(tello_image_process.ImageProcessService.KEY_FPS): False
    }))
    tello_center.register_service(tello_center.PreLoadService(tasks=[
        YoloService.preload
    ]))
    tello_center.register_service(tello_abs.TelloBackendService())  # 提供基础控制和数据
    tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    tello_center.register_service(tello_image_process.ImageProcessService())
    tello_center.register_service(YoloService())
    tello_center.start_all_service()
    tello_center.lock_loop()
