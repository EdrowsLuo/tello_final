# coding=utf-8
import time

import cv2
import numpy as np

from control import tello_center, tello_abs
import threading
"""
图形处理
"""


class ImageProcessService(tello_center.Service):
    name = 'image_service'

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
                h.process(img, state, showimg)
        finally:
            self.handler_lock.release()
            self.processing = False

    def main_loop(self):
        preid = self.id
        while True:
            if self.id > preid:
                preid = self.id
                showimg = None
                self.lock.acquire()
                try:
                    image, state = self.image, self.state
                    if image is not None and state is not None:
                        image = np.copy(image)
                        showimg = np.copy(image)
                    else:
                        continue
                finally:
                    self.lock.release()
                if image is not None and state is not None and showimg is not None:
                    self.do_process(image, state, showimg)
                    cv2.imshow(ImageProcessService.name, showimg)
                    cv2.waitKey(1)
            else:
                time.sleep(0.005)

    def start(self):
        def add_handler():
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
        pass

    def process(self, image, state, showimg):
        raise NotImplemented()


class DrawStateHandler(ImageHandler):
    def __init__(self):
        ImageHandler.__init__(self)

    def process(self, image, state, showimg):
        y = 14
        for s in state.raw:
            showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
            showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
            y += 14
