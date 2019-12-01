#!/usr/bin/env python
import sys

from direct.showbase.ShowBase import ShowBase

from panda3d.core import Geom, GeomNode
from panda3d.core import GeomVertexFormat, GeomVertexWriter, GeomVertexData
from panda3d.core import GeomTriangles
from panda3d.core import NodePath
from panda3d.core import PointLight
from panda3d.core import VBase4


class FooBarTriangle(ShowBase):
    def __init__(self):
        # Basics
        ShowBase.__init__(self)

        #base.disableMouse()
        base.setFrameRateMeter(True)

        self.accept("escape", sys.exit)
        self.camera.set_pos(-10, -10, 10)
        self.camera.look_at(0, 0, 0)

        # A light
        plight = PointLight("plight")
        plight.setColor(VBase4(1, 1, 1, 1))

        plnp = render.attachNewNode(plight)
        plnp.setPos(100, 100, 100)

        render.setLight(plnp)

        # Create the geometry
        vformat = GeomVertexFormat.getV3n3c4()

        vdata = GeomVertexData("Data", vformat, Geom.UHStatic)
        vdata.setNumRows(3)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        color = GeomVertexWriter(vdata, 'color')

        vertex.addData3f(100, 0, 0)
        normal.addData3f(0, 0, 1)
        color.addData4f(0, 1, 0, 1)

        vertex.addData3f(100, 100, 0)
        normal.addData3f(0, 0, 1)
        color.addData4f(0, 0, 1, 1)

        vertex.addData3f(0, 100, 0)
        normal.addData3f(0, 0, 1)
        color.addData4f(1, 0, 0, 1)

        prim = GeomTriangles(Geom.UHStatic)

        prim.addVertices(0, 1, 2)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode("GNode")
        node.addGeom(geom)

        nodePath = self.render.attachNewNode(node)


demo = FooBarTriangle()
if __name__ == '__main__':
    demo.run()
