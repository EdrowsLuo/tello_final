# coding=utf-8
import numpy as np
import copy


class Entity:

    def __init__(self, tags):
        self.tags = copy.copy(tags)

    @staticmethod
    def add_tag(tags, dic):
        t = {}
        if tags is not None:
            for key in tags:
                t[key] = tags[key]
        for key in dic:
            t[key] = dic[key]
        return t

    def ray_test(self, pos, direction):
        """
        射线检测，没有检测到返回None，检测到则返回碰撞位置
        :param pos:
        :param direction:
        """
        return None


class SurfaceEntity(Entity):

    class SurfaceData:
        def __init__(self, vec_x, vec_y):
            self.vec_x = vec_x
            self.vec_y = vec_y

    def __init__(self, tags):
        Entity.__init__(self, Entity.add_tag(tags, {'surface': True}))

    def ray_test(self, pos, direction):
        return None


class BoxEntity(Entity):
    class BoxData:
        def __init__(self, size, anchor):
            self.size = np.array(size, dtype=np.float64)
            self.anchor = np.array(anchor, dtype=np.float64)

    def __init__(self, size, anchor, tags=None):
        """
        :type tags: dict
        """
        self.data = BoxEntity.BoxData(size, anchor)
        self.position = np.array([0, 0, 0], dtype=np.float64)

        Entity.__init__(self, Entity.add_tag(tags, {'box': True}))


class World:

    def __init__(self):
        self.root = []


if __name__ == '__main__':
    pass
