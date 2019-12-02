import cv2
import numpy as np
import time
from mtellopy.tello import Tello
import socket
from tello_data import TelloData
import threading
import sl4p
import av


class MyTello:

    def __init__(self, drone, local_ip=''):
        """
        :type drone: Tello
        """
        self.logger = sl4p.Sl4p("my_tello", "1;33")
        self.drone = drone

        # state receive
        self.socket_state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.results = None
        self.latest_safe_state = None
        self.tello_data = None  # type: TelloData
        self.state_lock = threading.Lock()
        self.socket_state.bind((local_ip, 8890))
        self.receive_state_thread = threading.Thread(target=self._receive_state_thread)
        self.receive_state_thread.daemon = True
        self.receive_state_thread.start()

        # video
        self.image = None
        self.image_lock = threading.Lock()
        self.video_thread = threading.Thread(target=self._receive_video_thread)
        self.video_thread.daemon = True
        self.video_thread.start()

    def wait_until_video_done(self, timeout_s=15):
        start_time = time.time()
        while True:
            if self.get_frame() is not None:
                return True
            if time.time() - start_time > timeout_s:
                return False
            time.sleep(0.05)

    def get_frame(self):
        self.image_lock.acquire()
        try:
            return self.image
        finally:
            self.image_lock.release()

    def get_state(self):
        # type: () -> TelloData
        self.state_lock.acquire()
        try:
            return self.tello_data
        finally:
            self.state_lock.release()

    def _receive_state_thread(self):
        while True:
            try:
                state, ip = self.socket_state.recvfrom(1024)
                out = state.replace(';', ';\n')
                self.results = out.split()
                if not (self.results == 'ok'):
                    self.state_lock.acquire()
                    try:
                        s = TelloData("".join(self.results[0:8]))
                        if s.mid != -1:
                            self.latest_safe_state = s
                        self.tello_data = s
                    finally:
                        self.state_lock.release()
            except socket.error as exc:
                self.logger.error("Caught exception socket.error : %s" % exc)

    def _receive_video_thread(self):
        container = av.open(self.drone.get_video_stream())
        while True:
            for frame in container.decode(video=0):
                image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                if image is not None:
                    self.image_lock.acquire()
                    try:
                        self.image = image
                    finally:
                        self.image_lock.release()
