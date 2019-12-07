from control import tello_center, tello_world
from panda.panda_models import *
import threading
import time


class PandaService(tello_center.Service):
    name = 'panda_service'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.thread = None
        self.startup = None
        self.call_exit = False

    def start(self):
        tello_center.lock_loop = self.model_thread

    def on_request_exit(self):
        self.call_exit = True
        if self.startup is not None:
            self.startup.finalizeExit()

    def model_thread(self):
        startup = PandaStartup()
        self.startup = startup
        PLight(color=(1, 1, 1, 1), position=vec3(3, -1, 4), size=0.1).apply(startup.render)
        PLight(color=(1, 1, 1, 1), position=vec3(0, -1, 4), size=0.1).apply(startup.render)

        fog = Fog('Fog')
        fog.setColor(0.2, 0.2, 0.2)
        fog.setLinearRange(0, 30)
        startup.render.setFog(fog)

        size = nar([7, 7, 0.01])
        ModelBox("floorBox", nar([-0.5, -2, -size[2]]), size, (0.4, 0.4, 0.4, 1), startup.render)
        tello = ModelTello(startup.render)

        models = world_model()
        for data in models:
            if 'box' in data['tag']:
                model = data['model']  # type: CollideBox
                ModelBox(data['id'], model.org, model.size, (0.7, 0.7, 0.7, 1), startup.render)

        def spinCameraTask(task):
            self = startup
            angleDegrees = task.time*30.0
            # angleDegrees = 0
            angleRadians = angleDegrees*(pi/180.0)
            # self.camera.setPos(10*sin(angleRadians), -10.0*cos(angleRadians), 4)
            #self.camera.setHpr(angleDegrees, 0, 0)
            # self.camera.look_at(3, 0, 1.5)
            # self.plight.setPos(10*sin(angleRadians), -10.0*cos(angleRadians), 3)

            tello.setPos(3*sin(angleRadians), -3*cos(angleRadians), 1.6)
            tello.setDirection(angleDegrees, 0, 0)

            time.sleep(1.0/80.0)
            return Task.cont

        def exitTask(task):
            if self.call_exit:
                raise BaseException()

        startup.task_mgr.add(spinCameraTask, 'spinCameraTask')
        # startup.task_mgr.add(exitTask, 'exitTask')

        tello_center.input_exit_thread()
        try:
            startup.run()
        except SystemExit:
            tello_center.call_request_exit()



