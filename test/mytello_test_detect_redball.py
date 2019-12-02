# coding=utf-8
import cv2
import time
import numpy as np
from image_detecte.redball_detecter import *

from control.tello_abs import MyTello
import mtellopy.tello

def handler(event, sender, data, **args):
    """Drone events handler, for testing.  """
    drone_handler = sender
    if event is drone_handler.EVENT_FLIGHT_DATA:
        print(data)


def init_drone():
    """Drone initiation function for testing.  """
    drone = mtellopy.tello.Tello()

    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(60.0)

    except Exception as ex:
        print(ex)
        drone.quit()
        return None
    return drone


if __name__ == '__main__':
    myTello = MyTello(init_drone())
    myTello.wait_until_video_done()
    while True:
        img = myTello.get_frame()
        print(myTello.get_state())
        showimg = np.copy(img)
        if img is not None:
            view = 45/180.0*np.pi
            det = 10/180.0*np.pi
            x, y, w, h = find_red_ball(img)
            if x is not None:
                cv2.rectangle(showimg, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                _y, _dis_y, _det_y = solve_system(det, (np.pi - view)/2, view, 720, x, x + h, 10)

                pix_size = 1.0/w*10.0  # 单位像素对应的实际长度
                la_x = 480
                la_y = 168
                cx = x + w/2
                cy = y + h/2

                rh = _y(la_y) - _y(cy)  # (360 - cy) * pix_size
                # delta = 20
                # ry1 = _y(la_y)
                # ry2 = _y(la_y + delta)
                # scale = (ry2 - ry1) / delta
                ry = (cx - la_x)*pix_size  # (cx - 480) * pix_size
                # ry += (self.window_x - state.x)*np.tan(state.mpry[1]/180.0*np.pi)

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
            cv2.imshow("img", showimg)
            cv2.waitKey(1)
        time.sleep(0.018)
