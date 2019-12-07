# coding=utf-8
from control import tello_center, tello_abs
import sl4p
import threading
"""
保存Tello的姿态、位置
"""


class TelloBotStateService(tello_center.Service):
    name = 'tello_state'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.logger = sl4p.Sl4p(name=TelloBotStateService.name)
        self.reactive = tello_center.service_proxy_by_class(tello_abs.ReactiveImageAndStateService)

    def update_handler(self, img, state):
        pass

    def start(self):
        def add_handler():
            self.logger.info('add reactive handler')
            self.reactive.add_handler(self.update_handler)
        tello_center.async_wait_until_proxy_available(
            self.reactive,
            target=add_handler
        )

    def is_mid_lost(self):
        pass

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
