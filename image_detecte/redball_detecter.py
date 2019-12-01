# coding=utf-8
import cv2
import numpy as np


def find_red_ball(img):
    kernel_4 = np.ones((4, 4), np.uint8)  # 4x4的卷积核
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 创建mask
    mask = cv2.inRange(hsv, np.array([156, 100, 100]), np.array([180, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask = cv2.bitwise_or(mask, mask2)

    # 后处理mask
    erosion = cv2.erode(mask, kernel_4, iterations=1)
    erosion = cv2.erode(erosion, kernel_4, iterations=1)
    dilation = cv2.dilate(erosion, kernel_4, iterations=1)
    dilation = cv2.dilate(dilation, kernel_4, iterations=1)

    # 寻找轮廓
    v = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(v) == 2:
        contours, hierarchy = v
    elif len(v) == 3:
        _, contours, hierarchy = v
    else:
        return None, None, None, None

    area = []  # type: List[int]
    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    if len(area) == 0:
        return None, None, None, None
    max_idx = int(np.argmax(np.array(area)))
    c = np.mean(contours[max_idx], axis=0)
    dis = np.empty(shape=len(contours[max_idx]))
    for i in range(len(dis)):
        dis[i] = np.linalg.norm(c[0] - contours[max_idx][i])
    # print np.std(dis)
    if area[max_idx] < 1000 or np.std(dis) > 5:
        return None, None, None, None
    x, y, w, h = cv2.boundingRect(contours[max_idx])
    return x, y, w, h

