import os

import torch
import cv2
import sl4p
import locks
from mydetect.models import EfficientDet
import numpy as np
from mydetect.datasets import get_augumentation, VOC_CLASSES
import copy
from image_detecte.detect import DetectResult
from utils.utils import plot_one_box


class Detect(object):
    """
        dir_name: Folder or image_file
    """

    def __init__(self, threshold, weights=None, iou_threshold=0.3, num_class=6, network='efficientdet-d0', size_image=(512, 512)):
        super(Detect,  self).__init__()
        self.logger = sl4p.Sl4p('my_detect')
        if weights is None:
            main_dir = os.path.split(os.path.abspath(__file__))[0]
            weights = os.path.join(main_dir, 'weights/checkpoint_efficientdet-d0_61.pth')
        self.weights = weights
        self.size_image = size_image
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')
        self.transform = get_augumentation(phase='test')
        self.show_transform = get_augumentation(phase='show')
        if self.weights is not None:
            self.logger.info('Load pretrained Model')
            checkpoint = torch.load(
                self.weights, map_location=lambda storage, loc: storage)
            num_class = checkpoint['num_class']
            network = checkpoint['network']

        self.model = EfficientDet(
            num_classes=num_class, network=network,
            is_training=False, threshold=threshold, iou_threshold=iou_threshold
        )

        if self.weights is not None:
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()
        self.model.eval()

    def detect(self, img=None, file_name=None):
        if file_name is not None:
            img = cv2.imread(file_name)
        origin_img = copy.deepcopy(img)
        augmentation = self.transform(image=img)
        img = augmentation['image']
        img = img.to(self.device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            scores, classification, transformed_anchors = self.model(img)
            rt = []
            bboxes = list()
            labels = list()
            bbox_scores = list()
            colors = list()
            for j in range(scores.shape[0]):
                bbox = transformed_anchors[[j], :][0]
                x1 = int(bbox[0]*origin_img.shape[1]/self.size_image[1])
                y1 = int(bbox[1]*origin_img.shape[0]/self.size_image[0])
                x2 = int(bbox[2]*origin_img.shape[1]/self.size_image[1])
                y2 = int(bbox[3]*origin_img.shape[0]/self.size_image[0])
                bboxes.append([x1, y1, x2, y2])
                label_name = VOC_CLASSES[int(classification[[j]])]
                labels.append(label_name)
                score = np.around(
                    scores[[j]].cpu().numpy(), decimals=2)
                bbox_scores.append(int(score))
                rt.append(
                    DetectResult(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        object_conf=score,
                        class_conf=1,
                        class_name=label_name,
                        class_idx=int(classification[[j]]),
                        color=(100, 100, 100)
                    )
                )
            return rt

    def draw_result(self, img, det, show=False):
        if det is None or len(det) == 0:
            if show:
                locks.imshow('result', img)
            return
        for det_pack in det:
            xyxy = []
            for c in [det_pack.x1, det_pack.y1, det_pack.x2, det_pack.y2]:
                xyxy.append(c)
            conf = det_pack.class_conf
            label = '%s %.2f %.2f'%(det_pack.class_name, conf, det_pack.object_conf)
            plot_one_box(xyxy, img, label=label, color=det_pack.color)
        if show:
            locks.imshow('result', img)


if __name__ == '__main__':
    detobj = Detect()
    img = cv2.imread('D:\\ros\\final\\tello_final\\image_detecte\\data\\samples\\0.jpg')
    l = detobj.detect(img)
    for s in l:
        print(s)
    detobj.draw_result(img, l, show=True)
    input()
