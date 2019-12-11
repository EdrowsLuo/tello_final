# coding=utf-8
from control import tello_center, tello_world, tello_abs, tello_data, tello_image_process
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
        self.backend = tello_center.service_proxy_by_class(tello_abs.TelloBackendService)
        self.box = None

    def start(self):
        tello_center.lock_loop = self.model_thread

    def on_request_exit(self):
        tello_center.Service.on_request_exit(self)
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
                color = (0.7, 0.7, 0.7, 1)
                if 'item' in data['tag']:
                    color = (227.0/255, 149.0/255, 40.0/255, 1)
                    model = data['model_big']
                ModelBox(data['id'], model.org, model.size, color, startup.render)
            elif 'surface' in data['tag']:
                model = data['model_box']  # type: CollideBox
                ModelBox(data['id'], model.org, model.size, (98.0/255, 210.0/255, 250.0/255, 1), startup.render)

        size = vec3(0.05, 0.05, 0.05)
        box = ModelBox('collide_box', -size/2, size, (1,0, 0, 1), startup.render)
        self.box = box

        def spinCameraTask(task):
            #self = startup
            angleDegrees = task.time*30.0
            # angleDegrees = 0
            angleRadians = angleDegrees*(pi/180.0)

            if self.backend.available() and self.backend.drone.get_state() is not None:
                state = self.backend.drone.get_state()  # type: tello_data.TelloData
                if state.mid > 0:
                    pos = state.get_pos()
                    hpr = state.get_hpr() / np.pi * 180
                    tello.setPos(pos[0], pos[1], pos[2])
                    tello.setDirection(hpr[0], hpr[1], hpr[2])
                else:
                    pass
            else:
                tello.setPos(3*sin(angleRadians), -3*cos(angleRadians), 1.6)
                tello.setDirection(angleDegrees, 0, 0)

            pos = tello_center.get_preloaded(tello_image_process.FireDetector.PRELOAD_FIRE_POS)
            if pos is not None:
                box.setPos(pos[0], pos[1], pos[2])
            time.sleep(1.0/60.0)
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


if __name__ == '__main__':
    tello_center.register_service(tello_center.ConfigService(config={
        tello_abs.TelloBackendService.CONFIG_STOP: True,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: True
    }))
    tello_center.register_service(tello_center.PreLoadService(tasks=[
        tello_image_process.ImageProcessService.preload
    ]))
    tello_center.register_service(tello_abs.TelloBackendService())  # 提供基础控制和数据
    tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    tello_center.register_service(tello_image_process.ImageProcessService(handlers=[
        # tello_image_process.ProxyImageHandler(tello_image_process.FireDetector)
    ]))  # 提供图片处理
    tello_center.register_service(tello_world.WorldService())  # 世界模型，提供碰撞检测
    tello_center.register_service(PandaService())  # 提供3D模型预览
    tello_center.start_all_service()
    tello_center.lock_loop()
