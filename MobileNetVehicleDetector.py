import pafy
import cv2
import numpy as np
from datetime import datetime
import socket
import imageio.v3 as iio

from Detector import Detector
from Vehicle import Vehicle
from VehicleCache import VehicleCache

# url = "https://www.youtube.com/watch?v=9lzOzmFXrRA"  # NC
# url = "https://www.youtube.com/watch?v=fuuBpBQElv4" #NH
# url = "https://www.youtube.com/watch?v=vWaFYoZ5qyM" #Apex


url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"  # Tilton

DETECT_FRAME = 3


def resize_frame(frame):
    scale_percent = 40
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized


class MobileNetVehicleDetector:

    def __init__(self, video: str):
        self.env = socket.gethostname().startswith('Matthew')
        self.classifier = Detector(self.env)
        self.vehicles = {}
        self.trackers, self.anchors, self.distances = [], [], []
        self.granularity = 0
        self.cache = VehicleCache()
        if video.startswith('https'):
            video = pafy.new(video)
            best = video.getbest(preftype="mp4")
            self.capture = cv2.VideoCapture(best.url)
        self.initialize_anchors()
        self.distances = self.initialize_distances()
        self.ratios = self.calculate_ratios()
        self.fps = []
        self.frame_count = 0

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
        distances = []
        i = 0
        while i < self.granularity - 1:
            print('Distance between point {} and {} (ft):'.format(i, i + 1))
            distance = input()
            try:
                distance = float(distance)
                distances.append(distance)
                i += 1
            except:
                print('Invalid distance given')
            self.draw_anchors(frame)
            # self.draw_distances(distances, frame)
            cv2.imshow('image', frame)
            cv2.waitKey(20)
        return distances

    # def draw_distances(self, distances, frame):
    #     for (i, distance) in enumerate(distances):
    #         a, b = self.anchors[i], self.anchors[i + 1]
    #         text = '{} ft'.format(distance)
    #         cv2.putText(frame, text, ((a[0] + b[0]) // 2 - 10, (a[1] + b[1]) // 2),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

    def calculate_ratios(self):
        ratios = []
        for (i, distance) in enumerate(self.distances):
            diff = np.sqrt((self.anchors[i][0] - self.anchors[i + 1][0]) ** 2
                           + (self.anchors[i][1] - self.anchors[i + 1][1]) ** 2)
            ratios.append(distance / diff)
        return ratios

    def run(self):
        if self.env:
            while True:
                _, frame = self.capture.read()
                self.process(frame)
        else:
            for frame in iio.imiter("test_video.mp4", plugin="pyav"):
                self.process(frame)

    def process(self, frame):
        start = datetime.now().timestamp()
        frame = resize_frame(frame)
        if self.frame_count % DETECT_FRAME == 0:
            self.trackers = self.classifier.detect_vehicles(frame)
        else:
            vehicles = self.update_trackers(frame)
            undetected = self.cache.get_undetected()
            self.remove_undetected(undetected)
            self.update_positions(vehicles)
            self.evaluate_vehicles(vehicles)

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

        self.fps.append(1 / (datetime.now().timestamp() - start))
        if len(self.fps) == 1000:
            print("FPS: {}".format(np.mean(self.fps)))
            self.fps.clear()
        self.frame_count += 1

    def update_trackers(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = []
        for tracker in self.trackers:
            tracker.update(rgb_frame)
            p = tracker.get_position()
            x, y, fx, fy = int(p.left()), int(p.top()), int(p.right()), int(p.bottom())
            boxes.append((x, y, fx, fy))
        return self.cache.update(boxes)

    def remove_undetected(self, undetected):
        for vehicle_id in undetected:
            del self.vehicles[vehicle_id]

    def update_positions(self, vehicles):
        for (vehicle_id, vehicle) in self.vehicles.items():
            if vehicle.direction == 0 and vehicle_id in vehicles:
                vehicle.positions.append(vehicles[vehicle_id])

    def evaluate_vehicles(self, vehicles):
        for (vehicle_id, position) in vehicles.items():
            vehicle = self.vehicles.get(vehicle_id, None)
            if not vehicle:
                self.vehicles[vehicle_id] = Vehicle(position, len(self.distances))
            elif not vehicle.evaluated:
                if vehicle.direction == 0:
                    vehicle.estimate_direction()
                vehicle.update_position(position, self.anchors)
                if vehicle.passed_all_points:
                    speed = vehicle.estimate_speed(self.distances, self.ratios)
                    if speed != -1:
                        print('id: {}, speed {}'.format(vehicle_id, speed))


if __name__ == '__main__':
    detector = MobileNetVehicleDetector(url)
    detector.run()
