import threading
import cv2


cv_lock = threading.Lock()


def imshow(name, img, delay=1):
    cv_lock.acquire()
    try:
        cv2.imshow(name, img)
        cv2.waitKey(delay)
    finally:
        cv_lock.release()
