# coding=utf-8
import cv2
import time
import numpy as np
from image_detecte.redball_detecter import *
from control.tello_abs import MyTello
import mtellopy

def handler(event, sender, data, **args):
    """Drone events handler, for testing.  """
    drone_handler = sender
    if event is drone_handler.EVENT_FLIGHT_DATA:
        print(data)


def init_drone():
    """Drone initiation function for testing.  """
    drone = mtellopy.Tello()

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

    myTello.drone.takeoff()
    time.sleep(5)
    myTello.drone.up(10)
    time.sleep(3)
    myTello.drone.up(0)

    while True:
        data = myTello.get_state()
        img = myTello.get_frame()
        if img is not None:
            cv2.imshow("img", img)
            cv2.waitKey(1)
        if data is not None and data.mid > 0:
            myTello.drone.clockwise(-data.mpry[1])
        time.sleep(0.018)
