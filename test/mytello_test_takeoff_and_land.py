import cv2
import time

from control.tello_abs import MyTello
import mtellopy
import threading


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


def imshow_thread():
    while True:
        img = myTello.get_frame()
        if img is not None:
            cv2.imshow("img", img)
            cv2.waitKey(1)
        time.sleep(0.018)


if __name__ == '__main__':
    myTello = MyTello(init_drone())
    video = threading.Thread(target=imshow_thread)
    video.daemon = True
    video.start()
    myTello.wait_until_video_done()

    myTello.drone.takeoff()
    time.sleep(5)
    myTello.drone.clockwise(10)
    time.sleep(5)
    myTello.drone.land()
