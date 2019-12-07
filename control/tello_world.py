# coding=utf-8
"""
世界模型数据
"""


def collide_ray_window(ray_org, ray):
    """
    计算第一阶段射线检测红点位置
    """
    pass


def collide_box(ray_org, ray, hint=-1):
    """
    根据射线来测试对应的检测结果归属于哪个箱子，设置hint可以增加对应box的判断范围
    """
    pass


def set_test_collide_box_pos(x, y, z):
    """
    设置在panda3d里显示的对应碰撞点的位置
    """
    pass


def get_box_type(box_id):
    """
    获取对应box检测到的物件名称，没有检测到则返回None
    """
    pass


def set_box_type(box_id, item_type):
    """
    设置对应box里的种类
    """
    pass
