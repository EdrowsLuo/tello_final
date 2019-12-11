# coding=utf-8
from control import tello_center
"""
ros接口抽象
"""


class RosService(tello_center.Service):
    name = 'ros_service'

    def __init__(self):
        tello_center.Service.__init__(self)

    def can_takeoff(self):
        """
        是否允许起飞
        """
        return True

    def publish_takeoff(self):
        """
        发布takeoff指令
        """
        pass

    def publish_find_item(self, item_type, box_id):
        """
        发表寻找到了物件的信息
        """
        pass

    def publish_pass_fire(self):
        """
        表示通过了火焰
        """
        pass

    def publish_done(self):
        """
        表示降落
        """
        pass


