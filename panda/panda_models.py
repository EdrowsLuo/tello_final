import sys
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from math import pi, sin, cos

from panda3d.core import Geom, GeomNode
from panda3d.core import GeomVertexFormat, GeomVertexWriter, GeomVertexData
from panda3d.core import GeomTriangles
from panda3d.core import NodePath
from panda3d.core import PointLight, AmbientLight, Fog
from panda3d.core import VBase4
import numpy as np
import time
from world.world import *

def nar(d):
    return np.array(d, dtype=np.float64)


def norm(d):
    # type: (np.ndarray) -> np.ndarray
    return d / np.linalg.norm(d)


class PLight:
    def __init__(self, color, position, size):
        self.color = color
        self.position = position
        self.size = size
        self.path = None
        self.plnp = None

    def apply(self, render):
        plight = PointLight("plight")
        plight.setColor(VBase4(self.color[0], self.color[1], self.color[2], self.color[3]))

        self.plnp = render.attachNewNode(plight)
        self.plnp.setPos(self.position[0], self.position[1], self.position[2])

        render.setLight(self.plnp)

        builder = ModelBuilder()
        builder.add_box(nar([0, 0, 0]) - self.size / 2,
                        nar([self.size, 0, 0]), nar([0, self.size, 0]), nar([0, 0, self.size]),
                        self.color)

        self.path = render.attachNewNode(builder.build("plight_box"))
        self.path.setPos(self.position[0], self.position[1], self.position[2])

    def setPos(self, x, y, z):
        self.position[0:] = x, y, z
        self.plnp.setPos(self.position[0], self.position[1], self.position[2])
        self.path.setPos(self.position[0], self.position[1], self.position[2])


class ModelBox:

    def __init__(self, name, anchor, size, color, parent):
        self.name = name
        self.size = size
        self.anchor = anchor
        self.size = size
        builder = ModelBuilder()
        builder.add_box(anchor, nar([size[0], 0, 0]), nar([0, size[1], 0]), nar([0, 0, size[2]]), color)
        self.path = parent.attachNewNode(builder.build(name))

    def setPos(self, x, y, z):
        self.path.setPos(x, y, z)

    def setDirection(self, h, p, r):
        self.path.setHpr(h, p, r)


class ModelTello:

    def __init__(self, parent):
        self.name = 'tello'
        builder = ModelBuilder()
        size = vec3(0.2, 0.2, 0.05)
        builder.add_box(-size/2.0, nar([size[0], 0, 0]), nar([0, size[1], 0]), nar([0, 0, size[2]]), (0.1, 0.1, 0.1, 1))
        builder.add_box(vec3(-0.025, 0, 0.025), nar([0.05, 0, 0]), nar([0, 0.1, 0]), nar([0, 0, 0.015]), (1, 0, 0, 1))
        builder.add_box(vec3(-0.005, 0, -0.005), vec3(0.01, 0, 0), vec3(0, 6, 0), vec3(0, 0, 0.01), (0, 0, 1, 1))
        self.path = parent.attachNewNode(builder.build(self.name))
        builder = ModelBuilder()
        builder.add_box(vec3(-0.005, -0.005, 0), vec3(0.01, 0, 0), vec3(0, 0.01, 0), vec3(0, 0, -4), (0, 1, 0, 1))
        self.down_path = parent.attachNewNode(builder.build("tello_down"))

    def setPos(self, x, y, z):
        self.path.setPos(x, y, z)
        self.down_path.setPos(x, y, z)

    def setDirection(self, h, p, r):
        self.path.setHpr(h, p, r)

class ModelBuilder:

    def __init__(self):
        self.vformat = GeomVertexFormat.getV3n3c4()
        self.vdata = GeomVertexData("Data", self.vformat, Geom.UHStatic)
        self.vdata.setNumRows(3)
        self.vertex = GeomVertexWriter(self.vdata, 'vertex')
        self.normal = GeomVertexWriter(self.vdata, 'normal')
        self.color = GeomVertexWriter(self.vdata, 'color')
        self.prim = GeomTriangles(Geom.UHStatic)
        self.idx = 0

    def _add_point(self, v, n, c):
        self.vertex.addData3f(v[0], v[1], v[2])
        self.normal.addData3f(n[0], n[1], n[2])
        self.color.addData4f(c[0], c[1], c[2], c[3])
        self.idx += 1
        return self.idx - 1

    def add_triangle(self, v1, v2, v3, n, c):
        i0 = self._add_point(v1, n, c)
        i1 = self._add_point(v2, n, c)
        i2 = self._add_point(v3, n, c)
        self.prim.addVertices(i0, i1, i2)

    def add_rect(self, org, w, h, n, c):
        i0 = self._add_point(org, n, c)
        i1 = self._add_point(org + w, n, c)
        i2 = self._add_point(org + w + h, n, c)
        i3 = self._add_point(org + h, n, c)
        self.prim.addVertices(i0, i1, i2)
        self.prim.addVertices(i2, i3, i0)

    def add_box(self, org, w, h, l, c):
        self.add_rect(org, w, h, -norm(l), c)
        self.add_rect(org, w, l, -norm(h), c)
        self.add_rect(org, h, l, -norm(w), c)
        self.add_rect(org + w, l, h, norm(w), c)
        self.add_rect(org + h, l, w, norm(h), c)
        self.add_rect(org + l, h, w, norm(l), c)

    def build(self, name):
        self.prim.closePrimitive()

        geom = Geom(self.vdata)
        geom.addPrimitive(self.prim)

        node = GeomNode(name)
        node.addGeom(geom)
        return node


class PandaStartup(ShowBase):
    def __init__(self, ambient_light=(0.5, 0.5, 0.5, 1)):
        # Basics
        ShowBase.__init__(self)

        # self.disableMouse()
        # self.useDrive()
        self.setFrameRateMeter(True)
        self.backfaceCullingOff()
        self.accept("escape", sys.exit)

        alight = AmbientLight("alight")
        alight.setColor(VBase4(ambient_light[0], ambient_light[1], ambient_light[2], ambient_light[3]))

        self.render.setLight(self.render.attachNewNode(alight))


def test1():
    class ModelBuilderTest(ShowBase):
        def __init__(self):
            # Basics
            ShowBase.__init__(self)

            # self.disableMouse()
            # self.useDrive()
            self.setFrameRateMeter(True)
            self.backfaceCullingOff()

            self.accept("escape", sys.exit)
            # self.camera.set_pos(-10, -10, 10)
            # self.camera.look_at(-10, 0, 0)

            # A light
            '''
            plight = PointLight("plight")
            plight.setColor(VBase4(1, 1, 1, 1))

            plnp = self.render.attachNewNode(plight)
            plnp.setPos(10, 10, 50)

            self.render.setLight(plnp)
            '''

            self.plight = PLight(nar([1, 1, 1, 1]), nar([5, 5, 5]), 0.1)
            self.plight.apply(self.render)



            size = nar([5, 0.05, 0.05])
            self.xModel = ModelBox("floorBox", nar([0, -size[1]/2, -size[2]/2]), size, (1, 0, 0, 1), self.render)
            size = nar([0.05, 5, 0.05])
            self.yModel = ModelBox("floorBox", nar([-size[0]/2, 0, -size[2]/2]), size, (0, 1, 0, 1), self.render)
            size = nar([0.05, 0.05, 5])
            self.zModel = ModelBox("floorBox", nar([-size[0]/2, -size[1]/2, 0]), size, (0, 0, 1, 1), self.render)

            # Create the geometry
            self.redBox = ModelBox("redBox", nar([0, 0, 0]), nar([1, 1, 1]), (1, 0, 0, 1), self.render)
            self.redBox.setPos(0, 0, 0)

            size = nar([25, 25, 0.01])
            self.floorModel = ModelBox("floorBox", nar([0, 0, -size[2]]), size, (0.9, 0.9, 0.9, 1), self.render)

            size = nar([25, 0.1, 25])
            self.wallModel = ModelBox("wall", nar([0, 10, 0]), size, (1, 1, 1, 1), self.render)
            self.wall = CollideSurface(vec3(0, 10, 0), vec3(25, 0, 0), vec3(0, 0, 25), vec3(0, -1, 0))

            size = nar([0.1, 25, 0.1])
            self.rayModel = ModelBox("wall", nar([-0.05, 0, -0.05]), size, (1, 0, 0, 1), self.render)
            self.rayModel.setPos(10, 0, 10)

            size = nar([0.1, 0.1, 0.1])
            self.pointModel = ModelBox("wall", -size/2.0, size, (0, 0, 1, 1), self.render)

            self.task_mgr.add(self.spinCameraTask, "SpinCameraTask")

        # Define a procedure to move the camera.
        def spinCameraTask(self, task):
            angleDegrees = task.time * 30.0
            ray = nvec3(np.cos(angleDegrees/180), 2, np.sin(angleDegrees/180))
            point = self.wall.collide_ray(vec3(10, 0, 10), ray)
            if point is None:
                self.pointModel.setPos(0, 0, 0)
            else:
                self.pointModel.setPos(point[0][0], point[0][1], point[0][2])
            self.rayModel.setDirection(np.arctan(-ray[0] / ray[1]) / np.pi * 180, np.arctan(ray[2] / ray[1]) / np.pi * 180, 0)
            angleDegrees = 0
            angleRadians = angleDegrees*(pi/180.0)
            self.camera.setPos(40*sin(angleRadians), -40.0*cos(angleRadians), 3)
            self.camera.setHpr(angleDegrees, 0, 0)
            self.camera.look_at(0, 0, 3)
            #self.plight.setPos(10*sin(angleRadians), -10.0*cos(angleRadians), 3)
            time.sleep(1.0/80.0)

            return Task.cont

    #m = ModelBuilderTest()
    #m.run()


def test2():
    startup = PandaStartup()
    PLight(color=(1, 1, 1, 1), position=vec3(3, -1, 4), size=0.1).apply(startup.render)
    PLight(color=(1, 1, 1, 1), position=vec3(0, -1, 4), size=0.1).apply(startup.render)

    fog = Fog('Fog')
    fog.setColor(0.2, 0.2, 0.2)
    fog.setLinearRange(0, 30)
    startup.render.setFog(fog)

    size = nar([7, 7, 0.01])
    ModelBox("floorBox", nar([-0.5, -2, -size[2]]), size, (0.4, 0.4, 0.4, 1), startup.render)

    models = world_model()
    for data in models:
        if 'box' in data['tag']:
            model = data['model']  # type: CollideBox
            ModelBox(data['id'], model.org, model.size, (0.7, 0.7, 0.7, 1), startup.render)

    def spinCameraTask(task):
        self = startup
        angleDegrees = task.time*30.0
        angleRadians = angleDegrees*(pi/180.0)
        self.camera.setPos(40*sin(angleRadians), -40.0*cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        self.camera.look_at(0, 0, 3)
        # self.plight.setPos(10*sin(angleRadians), -10.0*cos(angleRadians), 3)
        time.sleep(1.0/80.0)
        return Task.cont

    # startup.task_mgr.add(spinCameraTask, 'spinCameraTask')
    startup.run()


if __name__ == '__main__':
    test2()
