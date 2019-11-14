# coding=utf-8
import time
from utils.utils import *

from detect import Detect
from tello_data import TelloData
import sl4p

def create_system(d, b, a):
    c = np.sin(d + a) * np.sin(b + a) / (np.sin(d + a + b) * np.sin(a))
    t = np.sin(a + b) / np.sin(d + a + b)
    k = np.sin(a + b) / np.sin(b)

    def _y(x):
        return t * x / (c * (1 - x) + x)

    def _det_y(x):
        tmp = c * (1 - x) + x
        return t * c / (tmp * tmp)

    def _dis_y(x):
        y = _y(x)
        return np.sqrt(y * y + k * k - 2 * y * k * np.cos(d + a))

    return _y, _dis_y, _det_y


def solve_system(d, b, a, x_scale, x1, x2, dis):
    x_scale = float(x_scale)
    _y, _dis_y, _det_y = create_system(d, b, a)
    y1, y2 = _y(x1 / x_scale), _y(x2 / x_scale)
    scale = dis / abs(y1 - y2)

    def __y(x):
        return scale * _y(x / x_scale)

    def __dis_y(x):
        return scale * _dis_y(x / x_scale)

    def __det_y(x):
        return scale * _det_y(x / x_scale)

    return __y, __dis_y, __det_y


def find_red_ball(img):
    kernel_4 = np.ones((4, 4), np.uint8)  # 4x4的卷积核
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 创建mask
    mask = cv2.inRange(hsv, np.array([156, 100, 100]), np.array([180, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask = cv2.bitwise_or(mask, mask2)

    # 后处理mask
    erosion = cv2.erode(mask, kernel_4, iterations=1)
    erosion = cv2.erode(erosion, kernel_4, iterations=1)
    dilation = cv2.dilate(erosion, kernel_4, iterations=1)
    dilation = cv2.dilate(dilation, kernel_4, iterations=1)

    # 寻找轮廓
    v = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(v) == 2:
        contours, hierarchy = v
    elif len(v) == 3:
        _, contours, hierarchy = v
    else:
        return None, None, None, None

    area = []
    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    if len(area) == 0:
        return None, None, None, None
    max_idx = np.argmax(np.array(area))

    if area[max_idx] < 1200:
        return None, None, None, None

    x, y, w, h = cv2.boundingRect(contours[max_idx])
    rate = area[max_idx] / (w * h)
    if rate < 0.4 or abs(w / float(h) - 1) > 0.7:
        return None, None, None, None
    return x, y, w, h


class TelloMain:
    class Stage:
        def __init__(self, id, fun):
            self.id = id
            self.fun = fun

    def __init__(self, tello):
        self.done = False
        self.logger = sl4p.Sl4p("tello_main", "1;36")
        self.start_time = None
        self.detector = None
        self.tello = tello
        self.stage = 0
        self.initial_done = False
        self.takeoff = False
        self.idx = 0

        self.ball_size = 10.0
        self.window_x = 180

        self.info_idx = 0

        self.latest_state = None
        self.fall_back_stage = None
        self.target_height = [130, 200, 160]
        self.current_search_times = 0
        self.detect_try_times = 0

    def initial(self):
        self.detector = Detect(0.5)
        self.initial_done = True
        self.print_info("x => 1", "initial done")

    def print_info(self, stage, msg):
        if self.info_idx == 0:
            self.info_idx = 1
            self.logger.info("\033[0;7;33m[stage:%s]\033[0m %s\033[0m" % (
                str(stage), msg.ljust(max(30, len(msg) + 2))))
        else:
            self.info_idx = 0
            self.logger.info("\033[0;7;37m[stage:%s]\033[0m %s\033[0m" % (
                str(stage), msg.ljust(max(30, len(msg) + 2))))

    def on_loop(self, state, img, showimg, do_draw=True, do_control=True):
        state = TelloData(state)
        if state.mid is not -1:
            if do_control:
                self.latest_state = state
        if self.stage == -1:
            if not do_control:
                return
            if self.takeoff:
                self.takeoff = False
                self.done = True
                self.print_info(1, "land!")
                self.tello.land()
        elif self.stage == -2:  # mid lost
            if not do_control:
                return
            if state.mid is not -1:
                next_stage = self.fall_back_stage or 2
                self.print_info("-2 => %d" % next_stage, "mid found, back to stage:%d" % next_stage)
                self.stage = next_stage
            elif self.latest_state is not None:
                if self.latest_state.x < 50:
                    self.print_info("-2", "move forward to find mid")
                    self.tello.move_forward(0.25)
                elif self.latest_state.x > 50:
                    self.print_info("-2", "move backward to find mid")
                    self.tello.move_backward(0.25)
                elif self.latest_state.y < 50:
                    self.print_info("-2", "move right to find mid")
                    self.tello.move_right(0.25)
                elif self.latest_state.x > 50:
                    self.print_info("-2", "move left to find mid")
                    self.tello.move_left(0.25)
                else:
                    self.print_info("-2", "move up to find mid")
                    self.tello.move_up(0.25)
        elif self.stage == 0:
            if not do_control:
                return
            if self.initial_done:
                self.print_info("0 => 1", "initial done")
                self.stage = 1
            else:
                return  # 尚未初始化，等待初始化
        elif self.stage == 1:  # 起飞并寻找定位毯
            if not do_control:
                return
            if self.takeoff:
                if state.mid == -1:  # 没有找到定位毯
                    self.print_info(1, "move up 25cm to find mid")
                    self.tello.move_up(0.25)
                else:  # 找到了定位毯
                    if state.y > 140 or state.y < 60:
                        self.print_info("1", "move into map (y)")
                        if state.y > 140:
                            dis = max(20, state.z - 100) / 100.0
                            self.tello.move_left(dis)
                        elif state.y < 60:
                            dis = max(20, 100 - state.z) / 100.0
                            self.tello.move_right(dis)
                        return
                    elif state.x > 140 or state.x < 60:
                        self.print_info("1", "move into map (x)")
                        if state.x > 140:
                            dis = max(20, state.x - 100) / 100.0
                            self.tello.move_backward(dis)
                        elif state.x < 60:
                            dis = max(20, 100 - state.x) / 100.0
                            self.tello.move_forward(dis)
                        return
                    else:
                        if self.fall_back_stage is not None:
                            next_stage = self.fall_back_stage or 2
                            self.print_info("1 => %d" % next_stage, "mid found, back to stage:%d" % next_stage)
                            self.stage = next_stage
                            self.fall_back_stage = None
                            return
                    self.print_info("1 => 2", "mid found")
                    self.stage = 2
            else:
                self.print_info(1, "takeoff")
                self.tello.takeoff()
                self.start_time = time.time()
                self.takeoff = True
        elif self.stage == 2:  # 调整姿态
            if not do_control:
                return
            if state.mid == -1:
                self.print_info("2 => 1", "mid lost!")
                self.stage = 1
                self.fall_back_stage = 2
            else:
                if abs(state.mpry[1]) > 6:  # 调整位姿
                    degree = abs(state.mpry[1])
                    degree = min(40, degree)
                    cw = state.mpry[1] < 0
                    self.print_info(2, "rotate %s %d" % (str(cw), degree))
                    if cw:
                        self.tello.rotate_cw(degree)
                    else:
                        self.tello.rotate_ccw(degree)
                    time.sleep(0.5)
                else:
                    self.print_info("2 => 3", "adjust pose done")
                    self.stage = 3
        elif self.stage == 3:  # 寻找着火点
            if state.mid == -1:
                if not do_control:
                    return
                self.print_info("3 => 1", "mid lost!")
                self.stage = 1
                self.fall_back_stage = 3
            elif state.y < 60:
                if not do_control:
                    return
                self.print_info(3, "move back to map (right)")
                self.tello.move_right(0.2)
            elif state.x < 40:
                if not do_control:
                    return
                self.print_info(3, "move back to map (forward)")
                self.tello.move_forward(0.2)
            elif state.x > 60:
                if not do_control:
                    return
                dis = max(min(state.x - 60, 35), 20) / 100.0
                self.print_info(3, "move back to find red point")
                self.tello.move_backward(dis)
            else:
                view = 45 / 180.0 * np.pi
                det = 10 / 180.0 * np.pi
                x, y, w, h = find_red_ball(img)
                if x is not None:  # 寻找到了着火点
                    if do_draw:
                        cv2.rectangle(showimg, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                    if abs(w / float(h) - 1) > 0.3:  # 只找到了部分园，判断在边界，进行调整
                        if not do_control:
                            return
                        if 720 - y - h < 20:
                            self.print_info(3, "move down to find full red point")
                            self.tello.move_down(0.2)
                        elif y < 20:
                            self.print_info(3, "move up to find full red point")
                            self.tello.move_up(0.2)
                        else:
                            # self.stage = -1
                            self.print_info("3 => -1", "bad red point rect! [%d, %d, %d, %d]" % (x, y, w, h))
                    else:  # 是一个完整的圆
                        _y, _dis_y, _det_y = solve_system(det, (np.pi - view) / 2, view, 720, x, x + h, 10)

                        pix_size = 1.0 / w * self.ball_size  # 单位像素对应的实际长度
                        la_x = 480
                        la_y = 168
                        cx = x + w / 2
                        cy = y + h / 2

                        rh = _y(la_y) - _y(cy)  # (360 - cy) * pix_size
                        delta = 20
                        ry1 = _y(la_y)
                        ry2 = _y(la_y + delta)
                        scale = (ry2 - ry1) / delta
                        ry = (cx - la_x) * pix_size  # (cx - 480) * pix_size

                        if do_draw:
                            cv2.line(showimg, (la_x, la_y), (cx, la_y), (0, 255, 0), thickness=2)
                            cv2.line(showimg, (cx, la_y), (cx, cy), (0, 255, 0), thickness=2)
                            s = "y offset: %.2fcm" % ry
                            cv2.putText(showimg, s, (la_x, la_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
                            cv2.putText(showimg, s, (la_x, la_y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                                        thickness=1)
                            s = "h offset: %.2fcm" % rh
                            cv2.putText(showimg, s, (cx, la_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
                            cv2.putText(showimg, s, (cx, la_y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255),
                                        thickness=1)

                        if not do_control:
                            return
                        if abs(ry) > 10:
                            dis = max(min(abs(ry), 40), 20) / 100.0
                            if ry < 0:
                                self.print_info(3, "adjust position to rush! (left)")
                                self.tello.move_left(dis)
                            else:
                                self.print_info(3, "adjust position to rush! (right)")
                                self.tello.move_right(dis)
                        elif abs(ry) > 7:
                            if ry < 0:
                                self.print_info(3, "adjust position to rush! (left)")
                                self.tello.move_left(0.3)
                            else:
                                self.print_info(3, "adjust position to rush! (right)")
                                self.tello.move_right(0.3)
                        elif rh > -10:
                            dis = max(min(abs(rh + 10), 40), 20) / 100.0
                            self.print_info(3, "adjust position to rush! (up)")
                            self.tello.move_up(dis)
                        elif rh < -20:
                            dis = max(min(abs(rh + 30), 40), 20) / 100.0
                            self.print_info(3, "adjust position to rush! (down)")
                            self.tello.move_down(dis)
                        else:  # 位置调整已经比较准确，rush
                            dis = ((self.window_x - state.x) + 30) / 100.0  # 估算到墙的距离
                            dis2 = (_dis_y(la_y) + 20) / 100.0
                            cv2.imshow("rush", showimg)
                            cv2.waitKey(1)
                            self.print_info(3, "rush %.2fm! (%.2f, %.2f)" % (dis, dis2, dis - dis2))
                            self.tello.move_forward(dis)
                            self.print_info("stage:3 => 4", "rush done! start detecting balls")
                            self.stage = 4
                else:  # 没有找到着火点
                    if not do_control:
                        return
                    if len(self.target_height) == 0:
                        self.print_info("3 => -1", "no more target height!")
                        self.stage = -1
                        return
                    else:
                        if self.current_search_times >= 5:
                            self.current_search_times = 0
                            self.print_info("3", "to new target height %d" % self.target_height.pop())
                            return
                        else:
                            self.current_search_times += 1
                    if abs(state.z - 160) > 20:  # 调整高度
                        dis = min(abs(state.z - 160), 40) / 100.0
                        self.print_info(3, "adjust height to 160 to find red point")
                        if state.z > 160:
                            self.tello.move_down(dis)
                        else:
                            self.tello.move_up(dis)
                    elif state.x > 80:  # 调整前后距离
                        dis = max(min(state.x - 80, 50), 20) / 100.0
                        self.print_info(3, "move back to find red point")
                        self.tello.move_backward(dis)
                    elif 90 < state.y < 110:
                        dis = max(min(120 - state.y, 50), 20) / 100.0
                        self.print_info(3, "from center move to right to find red point")
                        self.tello.move_right(dis)
                    elif state.y <= 90:
                        dis = max(min(90 - state.y, 50), 20) / 100.0
                        self.print_info(3, "from left move to center to find red point")
                        self.tello.move_right(dis)
                    else:
                        dis = max(min(state.y - 75, 60), 20) / 100.0
                        print("[tello][stage:3] move left to find red point")
                        self.tello.move_left(dis)
                        # self.stage = -1
                        # self.print_info("3 => -1", "failed to find red point")
        elif self.stage == 4:
            if not do_control:
                return
            result = self.detector.detect_ball(img)
            detected = False
            if len(result) > 0:
                for ball in result:
                    if "basket" in ball.class_name:
                        plot_one_box(
                            [ball.x1, ball.y1, ball.x2, ball.y2],
                            showimg,
                            color=ball.color,
                            label='%s %.2f' % (ball.class_name, ball.object_conf)
                        )
                        detected = True
            if detected:
                cv2.imshow("detect", showimg)
                cv2.waitKey(1)
                self.print_info(4, "ball detected")
                # self.tello.move_right(0.5)
                if state.mid != -1:
                    if state.z > 100:
                        dis = max(20, state.z - 100) / 100.0
                        self.tello.move_down(dis)
                    elif state.z < 50:
                        dis = max(20, 100 - state.z) / 100.0
                        self.tello.move_up(dis)
                    if state.y > 110:
                        dis = max(20, state.y - 90) / 100.0
                        self.tello.move_left(dis)
                    elif state.y < 80:
                        dis = max(20, 110 - state.y) / 100.0
                        self.tello.move_right(dis)
                    self.tello.move_forward(1.8)
                else:
                    self.tello.move_right(0.6)
                    self.tello.flip('l')
                self.print_info("stage:4 => -1", "process end, cost time %.2f" % (time.time() - self.start_time))
                self.stage = -1
            elif self.latest_state is not None and self.latest_state.z < 160:
                dis = random.randint(20, 40)
                self.print_info(4, "random up to find balls (up = %d)" % dis)
                self.tello.move_up(dis / 100.0)
            else:
                self.detect_try_times += 1
                if self.detect_try_times >= 6:
                    dis = random.randint(20, 40)
                    if random.randint(0, 10) > 3:
                        self.print_info(4, "random up to find balls (up = %d)" % dis)
                        self.tello.move_up(dis / 100.0)
                    else:
                        self.print_info(4, "random down to find balls (down = %d)" % dis)
                        self.tello.move_down(dis / 100.0)
                    self.detect_try_times = 0
                    return
                dis = random.randint(20, 40)
                if random.randint(0, 1) == 0:
                    self.print_info(4, "random move to find balls (left = %d)" % dis)
                    self.tello.move_left(dis / 100.0)
                else:
                    self.print_info(4, "random move to find balls (right = %d)" % dis)
                    self.tello.move_right(dis / 100.0)
        else:
            if not do_control:
                return
            self.print_info("%d => -1" % self.stage, "unknown stage")
            self.stage = -1


if __name__ == '__main__':
    print (find_red_ball(cv2.imread("./camera_screenshot_10.11.2019.png")))
