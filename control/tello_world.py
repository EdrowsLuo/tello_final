# coding=utf-8
from control import tello_center
from world.world import *
"""
世界模型数据
"""


class WorldService(tello_center.Service):
    name = 'world_service'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.model = None
        self.wallSurface = None  # type: CollideSurface

    def start(self):
        self.model = world_model()
        for m in self.model:
            if 'surface' in m['tag'] and 'wall' in m['tag']:
                self.wallSurface = m['model']

    def collide_ray_window(self, ray_org, ray):
        """
        计算第一阶段射线检测红点位置
        """
        return self.wallSurface.collide_ray(ray_org, ray)

    def collide_box(self, ray_org, ray, hint=-1):
        """
        根据射线来测试对应的检测结果归属于哪个箱子，设置hint可以增加对应box的判断范围
        """
        pass

    def set_test_collide_box_pos(self, x, y, z):
        """
        设置在panda3d里显示的对应碰撞点的位置
        """
        pass

    def get_box_type(self, box_id):
        """
        获取对应box检测到的物件名称，没有检测到则返回None
        """
        pass

    def set_box_type(self, box_id, item_type):
        """
        设置对应box里的种类
        """
        pass


