from control import tello_center
from control import tello_abs, tello_botstate, tello_image_process, tello_panda, tello_ros_reactive, tello_world

if __name__ == '__main__':
    tello_center.register_service(tello_center.ConfigService(config={
        tello_abs.TelloBackendService.CONFIG_STOP: True,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: False
    }))
    tello_center.register_service(tello_center.PreLoadService(tasks=[

    ]))
    tello_center.register_service(tello_abs.TelloBackendService())
    tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    tello_center.register_service(tello_botstate.TelloBotStateService())
    tello_center.register_service(tello_image_process.ImageProcessService())
    tello_center.register_service(tello_panda.PandaService())
    tello_center.register_service(tello_ros_reactive.RosService())
    tello_center.start_all_service()
    tello_center.lock_loop()
