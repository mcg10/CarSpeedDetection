import pafy
import cv2
import numpy as np
import dlib

#from Vehicle import Vehicle
from VehicleCache import VehicleCache

#url = "https://www.youtube.com/watch?v=9lzOzmFXrRA"  # NC
#url = "https://www.youtube.com/watch?v=fuuBpBQElv4" #NH
#url = "https://www.youtube.com/watch?v=vWaFYoZ5qyM" #Apex


url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"  # Tilton

prototext, model = 'MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel'

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

MIN_CONFIDENCE = .3
DETECT_FRAME = 3


def resize_frame(frame):
    scale_percent = 75
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


class MobileNetVehicleDetector:

    def __init__(self, video: str):
        self.classifier = cv2.dnn.readNetFromCaffe(prototext, model)
        self.vehicles = {}
        self.trackers, self.anchors, self.distances = [], [], []
        self.granularity = 0
        self.cache = VehicleCache()
        if video.startswith('https'):
            video = pafy.new(video)
            best = video.getbest(preftype="mp4")
            self.capture = cv2.VideoCapture(best.url)
        else:
            self.capture = cv2.VideoCapture(video)
        self.initialize_anchors()
        self.initialize_distances()

    def initialize_anchors(self):
        while self.granularity == 0:
            print("How many points of approximation would you like?")
            granularity = input()
            try:
                self.granularity = int(granularity)
            except:
                print("Invalid number")
        _, frame = self.capture.read()
        frame = resize_frame(frame)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.add_anchor)
        while len(self.anchors) < self.granularity:
            self.draw_anchors(frame)
            cv2.imshow('image', frame)
            cv2.waitKey(20)
        self.draw_anchors(frame)
        cv2.imshow('image', frame)
        cv2.waitKey(20)

    def add_anchor(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.anchors.append((x, y))

    def draw_anchors(self, frame):
        for (i, point) in enumerate(self.anchors):
            cv2.putText(frame, str(i), (point[0] - 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, (point[0], point[1]), 4, (255, 0, 0), -1)

    def initialize_distances(self):
        _, frame = self.capture.read()
        frame = resize_frame(frame)
        cv2.namedWindow('image')
        i = 0
        while i < self.granularity - 1:
            print('Distance between point {} and {} (ft):'.format(i, i + 1))
            distance = input()
            try:
                distance = float(distance)
                self.distances.append(distance)
                i += 1
            except:
                print('Invalid distance given')
            self.draw_anchors(frame)
            self.draw_distances(frame)
            cv2.imshow('image', frame)
            cv2.waitKey(20)

    def draw_distances(self, frame):
        for (i, distance) in enumerate(self.distances):
            a, b = self.anchors[i], self.anchors[i + 1]
            text = '{} ft'.format(distance)
            cv2.putText(frame, text, ((a[0] + b[0]) // 2 - 10, (a[1] + b[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    def run(self):
        frame_count = 0
        while True:
            _, frame = self.capture.read()
            frame = resize_frame(frame)
            if frame_count % DETECT_FRAME == 0:
                self.trackers.clear()
                detections = self.detect_objects(frame)
                height, width = frame.shape[:2]
                self.track_cars(detections, frame, height, width)
            else:
                vehicles = self.update_trackers(frame)
                #self.evaluate_vehicles(vehicles)

            for vehicle in self.cache.vehicles:
                centroid = self.cache.vehicles[vehicle]
                text = "ID {}".format(vehicle)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            for (i, point) in enumerate(self.anchors):
                cv2.putText(frame, str(i), (point[0] - 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(frame, (point[0], point[1]), 4, (255, 0, 0), -1)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            frame_count += 1

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
                                     ddepth=cv2.CV_8U)
        self.classifier.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5,
                                                                      127.5, 127.5])
        detections = self.classifier.forward()
        return detections

    def track_cars(self, detections, frame, height, width):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            if (CLASSES[idx] == 'car' or CLASSES[idx] == 'bus') and confidence >= MIN_CONFIDENCE:
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
        return self.cache.update(boxes)

    # def evaluate_vehicles(self, vehicles):
    #     for (vehicle_id, position) in vehicles:
    #         if vehicle_id not in self.vehicles:
    #             vehicle = Vehicle(position, len(self.an))



if __name__ == '__main__':
    detector = MobileNetVehicleDetector(url)
    detector.run()
