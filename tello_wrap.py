import threading
import cv2
import numpy as np
import tello_base as tello
from tello_base import Tello
from tello_main import TelloMain


class TelloWrap:
    def __init__(self, main, drone):
        # type: (TelloMain, Tello) -> None
        self.img_lock = threading.Lock()
        self.main = main
        self.drone = drone

    def _update_thread(self):
        while True:
            state = self.drone.read_state()
            if state is None or len(state) == 0:
                continue
            self.tello_state = "".join(state)

            frame = self.drone.read_frame()
            if frame is None or frame.size == 0:
                continue
            self.img_lock.acquire()
            try:
                self.img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            finally:
                self.img_lock.release()

    def get_img(self):
        self.img_lock.acquire()
        try:
            return self.img
        finally:
            self.img_lock.release()

    def main_loop(self):
        print("start")
        update_thread = threading.Thread(target=self._update_thread)
        update_thread.daemon = True
        update_thread.start()
        self.main.initial()
