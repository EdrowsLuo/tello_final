# coding=utf-8
from control import tello_center, tello_abs
import threading
"""
图形处理
"""


class ImageProcessService(tello_center.Service):
    name = 'image_service'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.reactive = tello_center.service_proxy_by_class(tello_abs.ReactiveImageAndStateService)
        self.image_thread = None
        self.processing = False
        self.image = None
        self.state = None

    def update_handler(self, img, state):
        pass

    def do_process(self):
        self.processing = True
        try:
            pass
        finally:
            self.processing = False

    def start(self):
        def add_handler():
            self.reactive.add_handler(self.update_handler)

        self.image_thread = threading.Thread()

        tello_center.async_wait_until_proxy_available(
            self.reactive,
            target=add_handler
        )

    def set_image_handler(self, handler):
        pass


class ImageHandler:
    def __init__(self):
        pass

    def process(self, image, state, show_img):
        raise NotImplemented()
