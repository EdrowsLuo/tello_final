# coding=utf-8
import cv2
import sl4p
from control import tello_center, tello_abs, tello_image_process
from image_detecte.detect import Detect, DetectImage, TaskLoop
from mydetect import detect
import threading
import time
import numpy as np
import locks


class YoloService(tello_center.Service):
    name = 'yolo_service'
    PRELOAD_YOLO_DETECTOR = 'yolo_detector'
    CONFIG_LOOP_DETECTION = 'yolo::loop_detection'
    CONFIG_DETECT_ON_MAIN_THREAD = 'yolo::CONFIG_DETECT_ON_MAIN_THREAD'
    CONFIG_USE_YOLO = 'yolo::CONFIG_USE_YOLO'
    CONFIG_YOLO_WEIGHTS = 'yolo::CONFIG_YOLO_WEIGHTS'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.logger = sl4p.Sl4p(YoloService.name)
        self.backend = tello_center.service_proxy_by_class(tello_abs.TelloBackendService)  # type: tello_abs.TelloBackendService
        self.task_loop = None

    @staticmethod
    def preload(service: tello_center.PreLoadService):
        if tello_center.get_config(YoloService.CONFIG_USE_YOLO, fallback=True):
            weights = tello_center.get_config(YoloService.CONFIG_YOLO_WEIGHTS, fallback='fire_new_1207.pt')
            service.put_loaded(YoloService.PRELOAD_YOLO_DETECTOR, Detect(0.1, weights=weights))
        else:
            service.put_loaded(YoloService.PRELOAD_YOLO_DETECTOR, detect.Detect(0.1))

    def post_detect(self, img):
        if tello_center.get_config(YoloService.CONFIG_DETECT_ON_MAIN_THREAD, fallback=True):
            task = DetectImage(tello_center.get_preloaded(YoloService.PRELOAD_YOLO_DETECTOR), img)
            return task.invoke_on(self.task_loop)
        else:
            return self.get_detector().detect(img)

    def detect(self, img, x=0, y=0, w=0, h=0):
        detector = tello_center.get_preloaded(YoloService.PRELOAD_YOLO_DETECTOR)  # type: Detect
        if detector is None:
            raise BaseException('detector not loaded')
        if w == 0 or h == 0:
            return self.post_detect(img)
        else:
            c = np.copy(img[y:y+h, x:x+w])
            locks.imshow('copy', c)
            results = self.post_detect(c)

            if results is not None:
                for r in results:
                    r.x1 += x
                    r.x2 += x
                    r.y1 += y
                    r.y2 += y
            else:
                results = []

            # for r in detector.detect(img) or []:
            #     results.append(r)
            if len(results) == 0:
                return None
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
                result = self.detect(img)
                #result = self.detect(img, 480, 100, 480, 360)
                detector.draw_result(img, result)
                locks.imshow('detect', img)
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

        if tello_center.get_config(YoloService.CONFIG_DETECT_ON_MAIN_THREAD, fallback=True):
            self.task_loop = TaskLoop(flag=self.flag)

            def override_lock_loop():
                tello_center.input_exit_thread(daemon=True)
                self.task_loop.start()
            tello_center.lock_loop = override_lock_loop


def main():
    tello_center.register_service(tello_center.ConfigService(config={
        tello_abs.TelloBackendService.CONFIG_STOP: True,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: True,

        YoloService.CONFIG_LOOP_DETECTION: True,
        YoloService.CONFIG_DETECT_ON_MAIN_THREAD: True,
        YoloService.CONFIG_YOLO_WEIGHTS: 'fire_89.pt',

        # FPS config
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_VIDEO_FPS): False,
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_STATE_FPS): False,
        tello_center.FpsRecoder.key(tello_image_process.ImageProcessService.KEY_FPS): False
    }))
    tello_center.register_service(tello_center.PreLoadService(tasks=[
        YoloService.preload
    ]))
    tello_center.register_service(tello_abs.TelloBackendService())  # 提供基础控制和数据
    # tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    # tello_center.register_service(tello_image_process.ImageProcessService())
    tello_center.register_service(YoloService())
    tello_center.start_all_service()
    tello_center.lock_loop()


if __name__ == '__main__':
    main()
