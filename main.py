import pafy
import cv2
import numpy as np
#from dlib import correlation_tracker, rectangle

#url = "https://www.youtube.com/watch?v=9lzOzmFXrRA" #NC
#url = "https://www.youtube.com/watch?v=fuuBpBQElv4" #NH
url = "https://www.youtube.com/watch?v=5_XSYlAfJZM" #Tilton

prototext, model = 'MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel'

classifier = cv2.dnn.readNetFromCaffe(prototext, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


class MobileNetVehicleDetector:

    def __init__(self, video: str):
        self.classifier = cv2.dnn.readNetFromCaffe(prototext, model)
        if video.startswith('https'):
            video = pafy.new(video)
            best = video.getbest(preftype="mp4")
            self.capture = cv2.VideoCapture(best.url)
        else:
            self.capture = cv2.VideoCapture(video)

    def run(self):
        while True:
            grabbed, frame = self.capture.read()
            frame = self.resize_frame(frame)
            detections = self.detect_objects(frame)
            height, width = frame.shape[:2]
            self.draw_bounding_boxes(detections, frame, height, width)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    def detect_objects(self, frame: np.ndarray):
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
                                     ddepth=cv2.CV_8U)
        classifier.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5,
                                                                 127.5, 127.5])
        detections = classifier.forward()
        return detections

    def draw_bounding_boxes(self, detections: np.ndarray, frame: np.ndarray, height: int,
                            width: int):
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            if (CLASSES[idx] == 'car' or CLASSES[idx] == 'bus') and confidence >= .3:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x, y, fx, fy = box.astype("int")
                cv2.rectangle(frame, (x, y), (fx, fy), (255, 0, 0), 4)

    def resize_frame(self, frame: np.ndarray):
        scale_percent = 40  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return resized



if __name__ == '__main__':
    detector = MobileNetVehicleDetector(url)
    detector.run()
