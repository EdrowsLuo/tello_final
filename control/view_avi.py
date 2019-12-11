# coding=utf-8
import cv2
import time

if __name__ == '__main__':
    cap = cv2.VideoCapture('./output.avi')

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            time.sleep(0.018)
            continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(18) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()