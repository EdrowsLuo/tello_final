# coding=utf-8
import threading
import time

import cv2
import numpy as np

import sl4p
from control import tello_center, tello_abs, tello_data, tello_yolo, tello_judge_client
from image_detecte.redball_detecter import *
from image_detecte import detect
from utils.drone_util import *


class Stage:
    def __init__(self, name, func_do=None, func_into=None, func_leave=None, args=None):
        self.name = name
        self.args = args
        self.func_do = func_do
        self.func_into = func_into
        self.func_leave = func_leave

    def on_into_stage(self):
        if self.func_into is not None:
            self.func_into()

    def do(self):
        if self.func_do is not None:
            if self.args is not None:
                self.func_do(self.args)
            else:
                self.func_do()

    def on_leave_stage(self):
        if self.func_leave is not None:
            self.func_leave()


class MainControl(tello_center.Service):
    name = 'main_control'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.logger = sl4p.Sl4p(MainControl.name)
        self.backend = tello_center.service_proxy_by_class(tello_abs.TelloBackendService)  # type: tello_abs.TelloBackendService
        self.yolo = tello_center.service_proxy_by_class(tello_yolo.YoloService)  # type: tello_yolo.YoloService
        self.judge = tello_center.service_proxy_by_class(tello_judge_client.JudgeClientService)  # type:tello_judge_client.JudgeClientService
        self.stage = None
        self.args = None

        self.stage_wait_for_start = Stage('wait_for_start', func_do=self.wait_for_start)
        self.stage_find_fire = Stage('find_fire', func_do=self.search_fire)
        self.found_fire = None
        self.search_min = True

        self.stage_go_to_step2_start_pos = Stage('step2', func_do=self.step2)

        self.stage_land = Stage('land', func_do=self.land)
        self.stage_idle = Stage('idle', func_do=self.idle)

    def jump_stage(self, stage, args=None):
        self.logger.info("jump stage: %s" % str(stage.name))
        if self.stage is not None:
            self.stage.on_leave_stage()
        self.stage = stage
        self.stage.args = args
        self.stage.on_into_stage()

    def wait_for_start(self):
        if self.backend.available():
            if self.backend.drone.has_takeoff:
                self.jump_stage(self.stage_find_fire)
                return
            if not self.backend.drone.has_takeoff:
                self.logger.info("takeoff")
                self.judge.server.takeoff()
                self.backend.drone.takeoff()
                # self.backend.drone.go(0, 0, 0.6)
                self.backend.drone.go(0.6, -1.0, 0.6)

                #self.backend.drone.go(0, 0, 0.6)
                #self.backend.drone.rotate_ccw(90)
                #look_at(self.backend, )
                #go_abs(self.backend.drone, self.backend.drone.get_state(), vec3(1.0, 0, 0))
                self.logger.info("takeoff done")
        else:
            time.sleep(0.01)
            return

    def land(self):
        if self.backend.drone.has_takeoff:
            self.backend.drone.land()
            self.judge.server.send_task_done()
        time.sleep(0.01)

    def idle(self):
        time.sleep(1)

    def on_found_fire(self, pos):
        self.found_fire = pos

    def search_fire(self):
        image, state = self.backend.drone.get_image_and_state()  # type: object, tello_data.TelloData
        if state.mid != -1 and False:
            if abs(state.mpry[1]) > 8 and not 150 < state.y + 50 < 250:
                if state.mpry[0] > 0:
                    self.backend.drone.rotate_ccw(state.mpry[0])
                else:
                    self.backend.drone.rotate_cw(-state.mpry[0])
                return
            if state.x < 30:
                self.backend.drone.move_forward(0.2)
                return
        x, y, w, h = find_red_ball(image)
        if x is not None:
            view = 45/180.0*np.pi
            det = 10/180.0*np.pi
            _y, _dis_y, _det_y = solve_system(det, (np.pi - view)/2, view, 720, x, x + h, 10)

            pix_size = 1.0/w*10  # 单位像素对应的实际长度
            la_x = 480
            la_y = 168
            cx = int(x + w/2)
            cy = int(y + h/2)

            rh = _y(la_y) - _y(cy)  # (360 - cy) * pix_size
            ry = (cx - la_x)*pix_size  # (cx - 480) * pix_size
            ry += (180 - (state.x if state.mid == -1 else 70))*np.tan(state.mpry[1]/180.0*np.pi)
            mz = 0
            my = 0
            if rh > -10:
                mz = max(min(abs(rh + 20), 40), 22)/100.0
                #up
            elif rh < -30:
                mz = -max(min(abs(rh + 20), 40), 22)/100.0
                #down

            if abs(ry) > 10:
                dis = max(min(abs(ry), 40), 22)/100.0
                if ry < 0:
                    my = -dis
                    #self.backend.drone.move_left(dis)
                else:
                    my = dis
                    #self.backend.drone.move_right(dis)

            if mz == 0 and my == 0:
                # 位置调整已经比较准确，rush
                dis = min(200, ((160 - (state.x if state.mid == -1 else 40)) + 30))/100.0  # 估算到墙的距离
                dis2 = (_dis_y(la_y) + 30)/100.0
                self.backend.drone.go(dis, ry / 100.0, (rh + 20) / 100.0)
                # clamp_abs(round(ry), 20, 40) / 100.0, clamp_abs(round(rh + 20), 20, 40) / 100.0)
                self.jump_stage(self.stage_go_to_step2_start_pos)
                self.judge.server.seen_fire()
            else:
                self.backend.drone.go(0, my, mz)
        else:
            if state.mid == -1:
                self.backend.drone.move_up(0.5)
                self.backend.drone.move_right(0.2)
            else:
                v1 = vec3(state.x, state.y, state.z)/100.0 - vec3(0.6, 0.9, 1.8)
                v2 = vec3(state.x, state.y, state.z)/100.0 - vec3(0.6, 2.9, 1.8)
                l1 = np.linalg.norm(v1)
                l2 = np.linalg.norm(v2)
                if self.search_min and l1 < l2:
                    goto(self.backend, 0.6, 0.9, 1.8, self.flag, tol=0.25)
                    look_at(self.backend, 10000, 0, 0, self.flag)
                else:
                    goto(self.backend, 0.6, 2.9, 1.8, self.flag, tol=0.25)
                    look_at(self.backend, 10000, 0, 0, self.flag)
                self.search_min = not self.search_min



            # elif state.x > 70:
            #    self.backend.drone.move_backward(max(state.x > 70, 25)/100.0)
            # elif abs(state.z - 180) > 15:
            #     if state.z > 180:
            #         self.backend.drone.move_down(max(min(state.z - 180, 50), 22)/100.0)
            #     else:
            #         self.backend.drone.move_up(max(min(180 - state.z, 50), 22)/100.0)
            # else:
            #     self.backend.drone.move_right(0.5)
            #     if 160 < state.y + 50 < 240:
            #        self.backend.drone.move_right(0.8)

    def step2(self):

        goto(self.backend, 2.7, 3.2, 2.05, self.flag, tol=0.30)

        look_at(self.backend, 10, 3.20, 2.1, self.flag)
        self.detect_object(5)

        goto(self.backend, 2.6, 2.35, 2.4, self.flag)

        look_at(self.backend, 7, 2.30, 2.1, self.flag)
        self.detect_object(3)

        goto(self.backend, 3.0, 1.50, 1.7, self.flag, tol=0.35)

        look_at(self.backend, 3.0, -2.0, 1.65, self.flag)
        if self.backend.drone.get_state().y - 0.50 < 0.5:
            self.backend.drone.move_backward(0.25)
        self.detect_object(1)

        # goto(self.backend, 4.6, 0.8, 1.7, self.flag)

        # look_at(self.backend, 9.1, 0.25, 1.7, self.flag)
        # self.detect_object(2, hint=(100, 0, 480 + 80, 360 + 60))
        # time.sleep(2)

        goto(self.backend, 4.6, 0.65, 1.6, self.flag, tol=0.30)

        look_at(self.backend, 1, 0.6, 1.7, self.flag)
        self.detect_object(1)

        goto(self.backend, 4.3, 0.9, 2.5, self.flag)

        look_at(self.backend, 4.3, 30, 2.3, self.flag)
        self.detect_object(3)

        goto(self.backend, 5.0, 1.0, 2.2, self.flag, tol=100)
        goto(self.backend, 5.5, 2.0, 2.45, self.flag, tol=0.35)

        look_at(self.backend, -10, 2.0, 2.4, self.flag)
        self.detect_object(3)

        goto(self.backend, 5.2,  3.4, 1.85, self.flag, tol=0.35)

        look_at(self.backend, -10, 3.6, 1.8, self.flag)
        self.detect_object(5)

        goto(self.backend, 4.4, 3.0, 2.45, self.flag, tol=0.3)

        look_at(self.backend, 4.3, -3, 2.45, self.flag)
        self.detect_object(3)

        goto(self.backend, 6.3, 3.2, 2.2, self.flag, tol=0.3)

        look_at(self.backend, 6.3, -2, 2.2, self.flag)
        self.detect_object(4)

        goto(self.backend, 7.4, 3.1, 2.2, self.flag, tol=1000)
        goto(self.backend, 7.5, 2.0, 2.2, self.flag, tol=0.3)

        look_at(self.backend, 1, 2.0, 2.2, self.flag)
        self.detect_object(4)

        goto(self.backend, 7.4, 0.8, 1.7, self.flag, tol=0.3)

        look_at(self.backend, 0, 0.25, 1.7, self.flag)
        self.detect_object(2)

        goto(self.backend, 7.3, 1.0, 2.2, self.flag, tol=0.3)
        
        look_at(self.backend, 6.3, 4.0, 2.2, self.flag)
        self.detect_object(4)

        goto(self.backend, 7.5, 1.7, 1.7, self.flag, tol=0.3)

        look_at(self.backend, 6.5, 0, 1.6, self.flag)
        self.detect_object(2, hint=(80, 60, 960 - 160, 720 - 120))

        goto(self.backend, 7.5, 2.0, 2.0, self.flag)
        look_at(self.backend, 1000, 0, 0, self.flag)
        self.backend.drone.move_forward(1.1)

        self.jump_stage(self.stage_land)

    def detect_object(self, idx, hint=None, count=10, skip=1, sleep_duration=0.05):
        if hint is None:
            hint = (0, 0, 0, 0)
        preimg = None
        frame = 0
        results = detect.ResultCollection()
        while self.flag() and count > 0:
            img, state = self.backend.drone.get_image_and_state()
            if img is preimg:
                time.sleep(sleep_duration)
            frame += 1
            if frame > skip:
                frame = 0
                count -= 1
                preimg = img
                result = self.yolo.detect(img, hint[0], hint[1], hint[2], hint[3])
                results.add_all_results(result)
                showim = np.copy(img)
                self.yolo.get_detector().draw_result(showim, result, show=True)

        results = results.get_result_collection()
        self.logger.info("%d %s" % (idx, str(results)))
        max_obj = find_most_possible_object(results)
        if max_obj is not None:
            self.logger.info("find object %d %s" % (idx, str(max_obj)))
            self.judge.put_chest_info(idx, max_obj['name'])

    def loop(self):
        if self.stage is not None:
            self.stage.do()

    def loop_thread(self):
        tello_center.wait_until_proxy_available(self.backend)
        self.backend.drone.wait_for_image_and_state()
        self.jump_stage(self.stage_wait_for_start)
        while True:
            self.loop()
            time.sleep(0.001)

    def start(self):
        t = threading.Thread(target=self.loop_thread)
        t.daemon = True
        t.start()
