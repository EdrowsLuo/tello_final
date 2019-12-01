import tellopy
import cv2
import av
import numpy as np
import time


def handler(event, sender, data, **args):
    """Drone events handler, for testing.  """
    drone_handler = sender
    if event is drone_handler.EVENT_FLIGHT_DATA:
        print(data)


def init_drone():
    """Drone initiation function for testing.  """
    drone = tellopy.Tello()

    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(60.0)

    except Exception as ex:
        print(ex)
        drone.quit()
        return None
    return drone


def main():
    drone = init_drone()
    container = av.open(drone.get_video_stream())
    while True:
        image = None
        for frame in container.decode(video=0):
            image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            if image is not None:
                cv2.imshow("camera", image)
                cv2.waitKey(1)
            # time.sleep(0.018)


if __name__ == '__main__':
    main()
