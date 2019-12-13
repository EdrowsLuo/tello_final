# coding=utf-8
import time

import cv2
import numpy as np

from control import tello_center, tello_abs, tello_data, tello_world
from image_detecte.redball_detecter import *
from image_detecte import redball_detecter
import threading, locks
"""
图形处理
"""


class ImageProcessService(tello_center.Service):
    name = 'image_service'
    KEY_FPS = 'image_process'

    def __init__(self, handlers=None):
        tello_center.Service.__init__(self)
        self.reactive = tello_center.service_proxy_by_class(tello_abs.ReactiveImageAndStateService)
        self.image_thread = None
        self.processing = False
        self.image = None
        self.state = None
        self.lock = threading.Lock()
        self.id = 0
        self.handler_lock = threading.Lock()
        self.handlers = []  # type: list[ImageHandler]
        self.handlers.append(DrawStateHandler())
        if handlers is not None:
            for h in handlers:
                self.handlers.append(h)

    @staticmethod
    def preload(service: tello_center.PreLoadService):
        service.put_loaded(FireDetector.key, FireDetector())

    def update_handler(self, img, state):
        self.lock.acquire()
        try:
            self.image = img
            self.state = state
            self.id += 1
        finally:
            self.lock.release()

    def do_process(self, img, state, showimg):
        self.processing = True
        self.handler_lock.acquire()
        try:
            for h in self.handlers:
                if h.enable:
                    h.process(img, state, showimg)
        finally:
            self.handler_lock.release()
            self.processing = False

    def main_loop(self):
        image_process_fps = tello_center.FpsRecoder(ImageProcessService.KEY_FPS)
        preid = self.id
        while True:
            if self.id > preid:
                image_process_fps.on_loop()
                preid = self.id
                showimg = None
                self.lock.acquire()
                try:
                    image, state = self.image, self.state
                finally:
                    self.lock.release()
                if image is not None and state is not None:
                    image = np.copy(image)
                    showimg = np.copy(image)
                else:
                    continue
                if image is not None and state is not None and showimg is not None:
                    self.do_process(image, state, showimg)
                    locks.imshow(ImageProcessService.name, showimg)
            else:
                time.sleep(0.001)

    def start(self):
        def add_handler():
            print('add handler')
            self.reactive.add_handler(self.update_handler)

        self.image_thread = threading.Thread(target=self.main_loop)
        self.image_thread.daemon = True
        self.image_thread.start()

        tello_center.async_wait_until_proxy_available(
            self.reactive,
            target=add_handler
        )

    def add_image_handler(self, handler):
        self.handler_lock.acquire()
        try:
            self.handlers.append(handler)
        finally:
            self.handler_lock.release()

    def remove_image_handler(self, handler):
        self.handler_lock.acquire()
        try:
            self.handlers.remove(handler)
        finally:
            self.handler_lock.release()


class ImageHandler:
    def __init__(self):
        self.enable = True

    def process(self, image, state, showimg):
        raise NotImplemented()


class ProxyImageHandler(ImageHandler):

    def __init__(self, klass):
        ImageHandler.__init__(self)
        self.key = klass.key

    def __getattribute__(self, item):
        if item == 'key':
            return object.__getattribute__(self, item)
        handler = tello_center.get_preloaded(self.key)
        if handler is None:
            if item == 'enable':
                return False
            return None
        return handler.__getattribute__(item)


class DrawStateHandler(ImageHandler):
    def __init__(self):
        ImageHandler.__init__(self)

    def process(self, image, state, showimg):
        y = 14
        for s in state.raw:
            showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
            showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
            y += 14


class FireDetector(ImageHandler):
    key = 'fire_detector'
    CONFIG_SHOW_POS_MAP = 'FireDetector::show_pos_map'
    PRELOAD_FIRE_POS = 'FireDetector::fire_pos'

    def __init__(self):
        ImageHandler.__init__(self)
        self.detectedPos = []
        self.world = tello_center.service_proxy_by_class(tello_world.WorldService)
        #self.panda = tello_center.service_proxy_by_class(tello_panda.PandaService)  # type: tello_panda.PandaService
        self.preload = tello_center.service_proxy_by_class(tello_center.PreLoadService)  # type:tello_center.PreLoadService
        self.pos_handler = None

    def process(self, image, state: tello_data.TelloData, showimg):
        x, y, w, h = redball_detecter.find_red_ball(image)
        if x is not None:
            cv2.rectangle(showimg, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            px = x + w / 2.0
            py = y + h / 2.0
            view = 45/180.0*np.pi
            det = 10/180.0*np.pi
            _y, _dis_y, _det_y = solve_system(det, (np.pi - view)/2, view, 720, x, x + h, 10)

            pix_size = 1.0/w*10  # 单位像素对应的实际长度
            la_x = 480
            la_y = 168
            cx = int(x + w/2)
            cy = int(y + h/2)

            rh = _y(la_y) - _y(cy)  # (360 - cy) * pix_size
            # delta = 20
            # ry1 = _y(la_y)
            # ry2 = _y(la_y + delta)
            # scale = (ry2 - ry1) / delta
            ry = (cx - la_x)*pix_size  # (cx - 480) * pix_size
            ry += (180 - state.x)*np.tan(state.mpry[1]/180.0*np.pi)
            cv2.line(showimg, (la_x, la_y), (cx, la_y), (0, 255, 0), thickness=2)
            cv2.line(showimg, (cx, la_y), (cx, cy), (0, 255, 0), thickness=2)
            s = "y offset: %.2fcm"%ry
            cv2.putText(showimg, s, (la_x, la_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
            cv2.putText(showimg, s, (la_x, la_y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                        thickness=1)
            s = "h offset: %.2fcm"%rh
            cv2.putText(showimg, s, (cx, la_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
            cv2.putText(showimg, s, (cx, la_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                        thickness=1)

            pos = state.get_pos()
            ray = state.get_ray(px, py)
            if self.world.available():
                pos = self.world.collide_ray_window(pos, ray)
                if pos is not None:
                    pos = pos[0]
                    # print(pos)
                    self.detectedPos.append(pos)
                    if self.pos_handler is not None:
                        self.pos_handler(pos)
                    s = '%.2f %.2f %.2f' % (pos[0], pos[1], pos[2])
                    la_x = 960 - 200
                    la_y = 720 - 30
                    cv2.putText(showimg, s, (la_x, la_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
                    cv2.putText(showimg, s, (la_x, la_y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                                thickness=1)
                    if tello_center.get_config(FireDetector.CONFIG_SHOW_POS_MAP, fallback=False):
                        fmap = np.array([206, 300, 3])
                        locks.imshow('fire_map', fmap)
                    self.preload.put_loaded(FireDetector.PRELOAD_FIRE_POS, np.copy(pos))
                elif self.pos_handler is not None:
                    self.pos_handler(None)


if __name__ == '__main__':
    tello_center.register_service(tello_center.ConfigService(config={
        tello_abs.TelloBackendService.CONFIG_STOP: True,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: True,

        # FPS config
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_VIDEO_FPS): False,
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_STATE_FPS): False,
        tello_center.FpsRecoder.key(ImageProcessService.KEY_FPS): False
    }))

    tello_center.register_service(tello_center.PreLoadService(tasks=[
        ImageProcessService.preload
    ]))
    tello_center.register_service(tello_abs.TelloBackendService())  # 提供基础控制和数据
    tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    tello_center.register_service(ImageProcessService(handlers=[
        # ProxyImageHandler(FireDetector)
    ]))  # 提供图片处理
    # tello_center.register_service(tello_world.WorldService())  # 世界模型，提供碰撞检测
    # tello_center.register_service(tello_panda.PandaService())  # 提供3D模型预览
    tello_center.start_all_service()
    tello_center.lock_loop()
