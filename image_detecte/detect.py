#!/usr/bin/env python
import time
from sys import platform

import sl4p
from models import *
import threading


class DetectResult:
    def __init__(self, x1, y1, x2, y2, object_conf, class_conf, class_name, color):
        self.x1, self.y1, self.x2, self.y2, self.object_conf, self.class_conf, self.class_name, self.color = \
            x1, y1, x2, y2, object_conf, class_conf, class_name, color

    def __str__(self):
        return "[(x1, y1, x1, y2) = (%d, %d, %d, %d), conf = (%.2f, %.2f), %s\ncolor = %s]" \
               %(self.x1, self.y1, self.x2, self.y2, self.object_conf, self.class_conf, self.class_name, str(self.color))


class Detect:
    def __init__(self, conf):
        # Initialize this once
        cfg = 'cfg/yolov3.cfg'
        weights = 'weights/ball.pt'
        output = 'data/output'
        img_size = 416
        self.conf_thres = conf

        device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
        torch.backends.cudnn.benchmark = False  # set False for reproducible results
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        # Initialize model
        if ONNX_EXPORT:
            s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
            model = Darknet(cfg, s)
        else:
            model = Darknet(cfg, img_size)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3, s[0], s[1]))
            torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
            return
        self.model = model
        self.device = device
        # Eval mode
        self.model.to(device).eval()

    def detect_ball(self, img):
        # type: (object) -> list[DetectResult]
        # Initialized  for every detection
        data = 'data/ball.data'
        output = 'data/output'
        img_size = 416
        nms_thres = 0.5
        save_txt = False
        save_images = False
        save_path = 'data/output/result.jpg'
        # Set Dataloader
        img0 = img  # BGR

        # Padded resize
        tmpresultimg = self.letterbox(img0, new_shape=img_size)
        img = tmpresultimg[0]
        # cv2.imshow("",img)
        # cv2.waitKey(0)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Get classes and colors
        classes = load_classes(parse_data_cfg(data)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        # Run inference
        # t0 = time.time()

        # Get detections

        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        # print("img.shape")
        # print(img.shape )
        pred, _ = self.model(img)
        det = non_max_suppression(pred.float(), self.conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        if det is None:
            return []
        if det.shape[0] <= 0:
            return []
        else:
            result = []
            for x in range(len(det)):
                result.append(DetectResult(det[x][0], det[x][1], det[x][2], det[x][3],
                                           det[x][4], det[x][5], classes[int(det[x][6])], colors[int(det[x][6])]))
            return result

    def letterbox(self, img, new_shape=416, color=(128, 128, 128), mode='auto'):
        # Resize a rectangular image to a 32 pixel multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            ratio = float(new_shape)/max(shape)
        else:
            ratio = max(new_shape)/max(shape)  # ratio  = new / old
        ratiow, ratioh = ratio, ratio
        new_unpad = (int(round(shape[1]*ratio)), int(round(shape[0]*ratio)))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if mode is 'auto':  # minimum rectangle
            dw = np.mod(new_shape - new_unpad[0], 32)/2  # width padding
            dh = np.mod(new_shape - new_unpad[1], 32)/2  # height padding
        elif mode is 'square':  # square
            dw = (new_shape - new_unpad[0])/2  # width padding
            dh = (new_shape - new_unpad[1])/2  # height padding
        elif mode is 'rect':  # square
            dw = (new_shape[1] - new_unpad[0])/2  # width padding
            dh = (new_shape[0] - new_unpad[1])/2  # height padding
        elif mode is 'scaleFill':
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape, new_shape)
            ratiow, ratioh = new_shape/shape[1], new_shape/shape[0]

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad,
                             interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return (img, ratiow, ratioh, dw, dh)


if __name__ == '__main__':
    logger = sl4p.Sl4p("__main__", "1")
    img = cv2.imread('data/samples/7.jpg')
    logger.info("start")
    detector = Detect(0.1)
    start = time.time()
    logger.info("start detect")
    result_obj = detector.detect_ball(img)
    end = time.time()
    logger.info("time: " + str(end - start) + "s")
    for r in result_obj:
        logger.info(str(r))

