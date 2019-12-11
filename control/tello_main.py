# coding=utf-8
from control import tello_center
from control import tello_abs, tello_botstate, tello_image_process, tello_panda, \
    tello_ros_reactive, tello_world, tello_control, tello_yolo

if __name__ == '__main__':
    tello_center.register_service(tello_center.ConfigService(config={
        # Tello backend config
        tello_abs.TelloBackendService.CONFIG_STOP: False,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: True,

        # Yolo config
        tello_yolo.YoloService.CONFIG_LOOP_DETECTION: False,

        # FPS config
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_VIDEO_FPS): False,
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_STATE_FPS): False,
        tello_center.FpsRecoder.key(tello_image_process.ImageProcessService.KEY_FPS): False
    }))
    tello_center.register_service(tello_center.PreLoadService(tasks=[
        tello_image_process.ImageProcessService.preload,
        tello_yolo.YoloService.preload
    ]))
    tello_center.register_service(tello_abs.TelloBackendService())  # 提供基础控制和数据
    tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    tello_center.register_service(tello_image_process.ImageProcessService(handlers=[
        #tello_image_process.ProxyImageHandler(tello_image_process.FireDetector)
    ]))  # 提供图片预览
    tello_center.register_service(tello_world.WorldService())  # 世界模型，提供碰撞检测
    tello_center.register_service(tello_yolo.YoloService())
    # tello_center.register_service(tello_panda.PandaService())  # 提供3D模型预览
    tello_center.register_service(tello_ros_reactive.RosService())  # 提供Ros指令
    tello_center.register_service(tello_control.MainControl())
    tello_center.start_all_service()
    tello_center.lock_loop()
