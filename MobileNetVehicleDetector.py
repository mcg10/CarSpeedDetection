import pafy
import cv2
import numpy as np
import dlib

# url = "https://www.youtube.com/watch?v=9lzOzmFXrRA" #NC
# url = "https://www.youtube.com/watch?v=fuuBpBQElv4" #NH
from VehicleCache import VehicleCache

url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"  # Tilton

prototext, model = 'MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel'

classifier = cv2.dnn.readNetFromCaffe(prototext, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


class MobileNetVehicleDetector:

    def __init__(self, video: str):
        self.classifier = cv2.dnn.readNetFromCaffe(prototext, model)
        self.trackers = []
        self.cache = VehicleCache()
        if video.startswith('https'):
            video = pafy.new(video)
            best = video.getbest(preftype="mp4")
            self.capture = cv2.VideoCapture(best.url)
        else:
            self.capture = cv2.VideoCapture(video)

    def run(self):
        frame_count = 0
        while True:
            grabbed, frame = self.capture.read()
            frame = self.resize_frame(frame)
            if frame_count % 4 == 0:
                self.trackers.clear()
                detections = self.detect_objects(frame)
                height, width = frame.shape[:2]
                self.track_cars(detections, frame, height, width)
            else:
                self.update_trackers(frame)

            for vehicle in self.cache.vehicles:
                centroid = self.cache.vehicles[vehicle]
                text = "ID {}".format(vehicle)
                cv2.putText(frame, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            frame_count += 1

    def detect_objects(self, frame: np.ndarray):
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
                                     ddepth=cv2.CV_8U)
        classifier.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5,
                                                                 127.5, 127.5])
        detections = classifier.forward()
        return detections

    def track_cars(self, detections: np.ndarray, frame: np.ndarray, height: int,
                   width: int):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            if (CLASSES[idx] == 'car' or CLASSES[idx] == 'bus') and confidence >= .3:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x, y, fx, fy = box.astype("int")
                tracker = dlib.correlation_tracker()
                rectangle = dlib.rectangle(x, y, fx, fy)
                tracker.start_track(rgb_frame, rectangle)
                self.trackers.append(tracker)

    def update_trackers(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = []
        for tracker in self.trackers:
            tracker.update(rgb_frame)
            p = tracker.get_position()
            x, y, fx, fy = int(p.left()), int(p.top()), int(p.right()), int(p.bottom())
            boxes.append((x, y, fx, fy))
        self.cache.update(boxes)

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
