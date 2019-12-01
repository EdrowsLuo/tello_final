#!/usr/bin/env python
import sys

from direct.showbase.ShowBase import ShowBase

from panda3d.core import Geom, GeomNode
from panda3d.core import GeomVertexFormat, GeomVertexWriter, GeomVertexData
from panda3d.core import GeomTriangles
from panda3d.core import NodePath
from panda3d.core import PointLight
from panda3d.core import VBase4
import numpy as np

def add_rect(idx, org, v1, v2, c, vertex, normal, color, prim):
    v = org
    vertex.addData3f(v[0], v[1], v[2])
    normal.addData3f(0, 0, 1)
    color.addData4f(c[0], c[1], c[2], c[3])

    v = org + v1
    vertex.addData3f(v[0], v[1], v[2])
    normal.addData3f(0, 0, 1)
    color.addData4f(c[0], c[1], c[2], c[3])

    v = org + v1 + v2
    vertex.addData3f(v[0], v[1], v[2])
    normal.addData3f(0, 0, 1)
    color.addData4f(c[0], c[1], c[2], c[3])

    v = org + v2
    vertex.addData3f(v[0], v[1], v[2])
    normal.addData3f(0, 0, 1)
    color.addData4f(c[0], c[1], c[2], c[3])

    prim.addVertices(4 * idx + 0, 4 * idx + 1, 4 * idx + 2)
    # prim.addVertices(4 * idx + 2, 4 * idx + 1, 4 * idx + 0)
    prim.addVertices(4 * idx + 2, 4 * idx + 3, 4 * idx + 0)
    # prim.addVertices(4 * idx + 0, 4 * idx + 3, 4 * idx + 2)


class FooBarTriangle(ShowBase):
    def __init__(self):
        # Basics
        ShowBase.__init__(self)

        # self.disableMouse()
        self.setFrameRateMeter(True)
        self.backfaceCullingOff()

        self.accept("escape", sys.exit)
        self.camera.set_pos(-10, -10, 10)
        self.camera.look_at(-10, 0, 0)

        # A light
        plight = PointLight("plight")
        plight.setColor(VBase4(1, 1, 1, 1))

        plnp = self.render.attachNewNode(plight)
        plnp.setPos(10, 10, 50)

        self.render.setLight(plnp)

        # Create the geometry
        vformat = GeomVertexFormat.getV3n3c4()

        vdata = GeomVertexData("Data", vformat, Geom.UHStatic)
        vdata.setNumRows(3)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')

        '''
        vertex.addData3f(0, 0, 0)
        normal.addData3f(0, 0, 1)
        color.addData4f(0, 0, 1, 1)

        vertex.addData3f(0, 150, 0)
        normal.addData3f(0, 0, 1)
        color.addData4f(0, 1, 0, 1)

        vertex.addData3f(10, 150, 0)
        normal.addData3f(0, 0, 1)
        color.addData4f(1, 0, 0, 1)

        vertex.addData3f(10, 0, 0)
        normal.addData3f(0, 0, 1)
        color.addData4f(0, 0, 0, 1)
        '''
        prim = GeomTriangles(Geom.UHStatic)

        '''
        prim.addVertices(0, 1, 2)
        prim.addVertices(2, 1, 0)
        prim.addVertices(2, 3, 0)
        prim.addVertices(0, 3, 2)
        '''
        org = np.array([0, 0, 0])
        add_rect(0, org, np.array([1, 0, 0]), np.array([0, 1, 0]), (1, 1, 1, 1), vertex, normal, color, prim)
        add_rect(1, org, np.array([-0.05, 0, 0]), np.array([0, 2, 0]), (0, 0, 1, 1), vertex, normal, color, prim)
        add_rect(2, org, np.array([2, 0, 0]), np.array([0, -0.05, 0]), (0, 1, 0, 1), vertex, normal, color, prim)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode("GNode")
        node.addGeom(geom)

        nodePath = self.render.attachNewNode(node)


if __name__ == '__main__':
    demo = FooBarTriangle()
    demo.run()
