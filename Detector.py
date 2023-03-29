import cv2
import numpy as np
import dlib

prototext, model = 'MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel'

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

MIN_CONFIDENCE = .3


class Detector:

    def __init__(self):
        self.detector = cv2.dnn.readNetFromCaffe(prototext, model)

    def detect_vehicles(self, frame):
        detections = self.detect_objects(frame)
        height, width = frame.shape[:2]
        return self.detect_all_cars(detections, frame, height, width)

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
                                     ddepth=cv2.CV_8U)
        self.detector.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5,
                                                                      127.5, 127.5])
        detections = self.detector.forward()
        return detections

    def detect_all_cars(self, detections, frame, height, width):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        trackers = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            if (CLASSES[idx] == 'car' or CLASSES[idx] == 'bus') and confidence >= MIN_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x, y, fx, fy = box.astype("int")
                tracker = dlib.correlation_tracker()
                rectangle = dlib.rectangle(x, y, fx, fy)
                tracker.start_track(rgb_frame, rectangle)
                trackers.append(tracker)
        return trackers

