from sys import path
path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

from control import tello_center, tello_abs, tello_image_process, tello_judge_client, tello_imshow, \
    tello_control
from control.tello_main import main
from control import tello_yolo

if __name__ == '__main__':
    tello_center.register_service(tello_center.ConfigService(config={
        # main config
        tello_center.ConfigService.CONFIG_DEBUG: True,

        # Tello backend config
        tello_abs.TelloBackendService.CONFIG_STOP: False,
        tello_abs.TelloBackendService.CONFIG_AUTO_WAIT_FOR_START_IMAGE_AND_STATE: True,

        # Yolo config
        tello_yolo.YoloService.CONFIG_LOOP_DETECTION: False,
        tello_yolo.YoloService.CONFIG_DETECT_ON_MAIN_THREAD: True,
        tello_yolo.YoloService.CONFIG_USE_YOLO: False,

        # FPS config
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_VIDEO_FPS): False,
        tello_center.FpsRecoder.key(tello_abs.MyTello.KEY_STATE_FPS): False,
        tello_center.FpsRecoder.key(tello_image_process.ImageProcessService.KEY_FPS): False,
    }))

    tello_center.register_service(tello_center.PreLoadService(tasks=[
        tello_yolo.YoloService.preload
    ]))

    tello_center.register_service(tello_imshow.ImshowService())
    tello_center.register_service(tello_abs.TelloBackendService())  # 提供基础控制和数据
    tello_center.register_service(tello_abs.ReactiveImageAndStateService())
    if tello_center.debug():
        tello_center.register_service(tello_image_process.ImageProcessService(handlers=[
            # tello_image_process.ProxyImageHandler(tello_image_process.FireDetector)
        ]))  # 提供图片预览
    tello_center.register_service(tello_judge_client.JudgeServerOverHttp())
    tello_center.register_service(tello_judge_client.JudgeClientService())
    #tello_center.register_service(tello_world.WorldService())  # 世界模型，提供碰撞检测
    # tello_center.register_service(tello_panda.PandaService())  # 提供3D模型预览
    tello_center.register_service(tello_yolo.YoloService())
    tello_center.register_service(tello_control.MainControl())
    tello_center.start_all_service()
    tello_center.lock_loop()

