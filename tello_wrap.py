import threading
import time

import cv2
import numpy as np
import tello_base as tello
from tello_base import Tello
from tello_data import TelloData
from tello_main import TelloMain
import sl4p

class TelloWrap:
    def __init__(self, main, drone):
        # type: (TelloMain, Tello) -> None
        self.update_lock = threading.Lock()
        self.main = main
        self.drone = drone
        self.logger = sl4p.Sl4p("main", "1;31")
        self.logger.LOG_LEVEL_MSG_STYLE[sl4p.LOG_LEVEL_INFO] = "1;31"
        self.tello_state = None
        self.img = None

    def _update_thread(self):
        while True:
            state = self.drone.read_state()
            if state is None or len(state) == 0:
                continue

            frame = self.drone.read_frame()
            if frame is None or frame.size == 0:
                continue
            self.update_lock.acquire()
            try:
                self.tello_state = "".join(state)
                self.img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            finally:
                self.update_lock.release()
            time.sleep(0.02)

    def _update_camera_thread(self):
        while True:
            state, img = self.get_data()
            if state is not None and img is not None:
                showimg = np.copy(img)
                if self.main.initial_done:
                    self.main.on_loop(state, img, showimg, do_control=False, do_draw=True)
                y = 14
                for s in str(state).split(";"):
                    showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
                    showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
                    y += 14
                cv2.imshow("camera", showimg)
                cv2.waitKey(1)
            time.sleep(0.01)

    def _control_thread(self):
        while True:
            if self.main.done:
                break
            state, img = self.get_data()
            if state is not None and img is not None:
                showimg = np.copy(img)
                self.main.on_loop(state, img, showimg, do_control=True, do_draw=True)
            time.sleep(0.01)

    def get_data(self):
        # type: () -> (str, object)
        self.update_lock.acquire()
        try:
            self.tello_state = "mid:1;x:50;y:80;z:-160;mpry:0,0,0;"
            return self.tello_state, self.img
        finally:
            self.update_lock.release()

    def main_loop(self):
        self.logger.info("start update thread")
        update_thread = threading.Thread(target=self._update_thread)
        update_thread.daemon = True
        update_thread.start()
        self.logger.info("start camera thread")
        camera_thread = threading.Thread(target=self._update_camera_thread)
        camera_thread.daemon = True
        camera_thread.start()
        self.logger.info("initial TelloMain")
        self.main.initial()
        self.logger.info("start control thread (main)")
        try:
            self._control_thread()
        except KeyboardInterrupt:
            self.logger.error("Ctrl-C detected, force land!")
            self.drone.land()
        except tello.TimeoutException as e:
            self.logger.error("Timeout: %s" % str(e))
            self.drone.land()

def filter_msg(msg):
    if "ok" in msg or "Done" in msg:
        return False
    else:
        return True


if __name__ == '__main__':
    drone = tello.Tello('', 8888)
    drone.do_print_info = True
    drone.filter = filter_msg
    drone.stop = True
    telloMain = TelloMain(drone)
    telloWrap = TelloWrap(telloMain, drone)
    telloWrap.main_loop()
