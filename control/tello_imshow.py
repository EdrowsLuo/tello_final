from control import tello_center
from image_detecte.detect import TaskLoop, Task
import locks
import numpy as np
import cv2
import sl4p


class ImshowTask(Task):
    def __init__(self, name, img, docopy=False):
        Task.__init__(self, self.show)
        self.name = name
        if docopy:
            img = np.copy(img)
        self.img = img

    def show(self):
        cv2.imshow(self.name, self.img)
        cv2.waitKey(1)


class ImshowService(tello_center.Service):
    name = 'imshow_service'

    def __init__(self):
        tello_center.Service.__init__(self)
        self.logger = sl4p.Sl4p('imshow')
        self.looper = None

    def imshow(self, name, img, docopy=False):
        # self.logger.info('imshow %s ' % name)
        t = ImshowTask(name, img, docopy=docopy)
        self.looper.add_task(t)

    def start(self):
        self.looper = TaskLoop(flag=self.flag)
        self.looper.start_async()
        locks.imshow = self.imshow


