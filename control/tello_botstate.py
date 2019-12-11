# coding=utf-8
import time

import cv2

from control import tello_center, tello_abs, tello_data
import sl4p
import threading

from image_detecte.detect import Detect

"""
保存Tello的姿态、位置
"""


class TelloBotStateService(tello_center.Service):
    name = 'tello_state'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.logger = sl4p.Sl4p(name=TelloBotStateService.name)
        self.reactive = tello_center.service_proxy_by_class(tello_abs.ReactiveImageAndStateService)
        self.update_lock = threading.Lock()
        self.state = None  # type: tello_data.TelloData

    def update_handler(self, img, state):
        self.update_lock.acquire()
        try:
            self.state = state
        finally:
            self.update_lock.release()

    def start(self):
        def add_handler():
            self.logger.info('add reactive handler')
            self.reactive.add_handler(self.update_handler)
        tello_center.async_wait_until_proxy_available(
            self.reactive,
            target=add_handler
        )

    def get_pos(self):
        """
        获取tello的姿态
        """
        pass

    def get_hpr(self):
        pass

    def get_ray(self, pixel_x, pixel_y):
        """
        根据当前tello姿态计算对应像素点对应射线检测数据
        """
        pass

    def get_look_at(self):
        """
        返回无人机正面方向
        """
        pass


if __name__ == '__main__':
    logger = sl4p.Sl4p("__main__", "1")
    img = cv2.imread('../image_detecte/data/samples/0.jpg')
    logger.info("start")
    detector = Detect(0.1)
    start = time.time()
    logger.info("start detect")
    result_obj = detector.detect(img)
    end = time.time()
    logger.info("time: " + str(end - start) + "s")
    for r in result_obj:
        logger.info(str(r))
