#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import tello_base as tello
import view_saved_img
from tello_main import TelloMain


def filter_msg(msg):
    if "ok" in msg or "Done" in msg or "imu" in msg or "joystick" in msg:
        return False
    else:
        return True


if __name__ == '__main__':
    drone = tello.Tello('', 8888)

    drone.do_print_info = True
    drone.filter = filter_msg
    print("start")
    telloMain = TelloMain(drone)
    telloMain.initial()

    # drone.stop = True
    try:
        while True:

            state = drone.read_state()
            if state is None or len(state) == 0:
                continue
            tello_state = "".join(state)

            frame = drone.read_frame()
            if frame is None or frame.size == 0:
                continue
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            showimg = np.copy(img)
            y = 14
            for s in state:
                showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness=5)
                showimg = cv2.putText(showimg, s, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
                y += 14

            try:
                telloMain.on_loop(tello_state, img, showimg)
                pass
            except tello.TimeoutException:
                pass

            cv2.imshow("camera", showimg)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        telloMain.print_info("outer", "KeyboardInterrupt(land)")
        drone.land()
    except tello.TimeoutException:
        telloMain.print_info("outer", "TimeoutException(land)")
        drone.land()
    # view_saved_img.display_saved_imgs()

