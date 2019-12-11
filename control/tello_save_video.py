# coding=utf-8
import threading

import cv2

from control import tello_center, tello_abs, tello_ros_reactive, tello_image_process


class SaveVideoService(tello_center.Service):
    name = 'save_video'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.reactive = tello_center.service_proxy_by_class(tello_abs.ReactiveImageAndStateService)
        self.frame = None
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 60.0, (960, 720))

    def on_new_frame(self, img, state):
        if self.frame is None or self.frame is not img:
            self.frame = img
            cv2.imshow('frame', self.frame)
            cv2.waitKey(1)
            self.out.write(self.frame)

    def start(self):
        def add_handler():
            print('add handler')

            self.reactive.add_handler(self.on_new_frame)

        tello_center.async_wait_until_proxy_available(
            self.reactive,
            target=add_handler
        )

    def on_request_exit(self):
        tello_center.Service.on_request_exit(self)
        self.out.release()


if __name__ == '__main__':
    tello_center.register_service(tello_center.ConfigService(config={
        tello_abs.TelloBackendService.CONFIG_STOP: True,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: True
    }))
    tello_center.register_service(tello_center.PreLoadService(tasks=[

    ]))
    tello_center.register_service(tello_abs.TelloBackendService())  # 提供基础控制和数据
    tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    tello_center.register_service(SaveVideoService())
    # tello_center.register_service(tello_image_process.ImageProcessService())
    tello_center.start_all_service()
    tello_center.lock_loop()
