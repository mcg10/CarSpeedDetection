from datetime import datetime
import numpy as np


class Vehicle:

    def __init__(self, position, points):
        self.anchors = {}
        self.positions = [position]
        for i in range(points):
            self.anchors[i] = None
        self.direction = 0
        self.evaluated = False
        self.passed_all_points = False

    def estimate_direction(self):
        average = np.mean([p[0] for p in self.positions])
        self.direction = average - self.positions[0][0]

    def update_position(self, position, points):
        p = points.copy()
        backwards = False
        if self.direction < 0:
            p = p[::-1]
            backwards = True
        for (i, point) in enumerate(p):
            if not self.anchors.get(i, None):
                if (not backwards and position[0] > point[0]) or (backwards and position[0] < point[0]):
                    self.anchors[i] = (datetime.now().timestamp(), position)
                    if point == p[-1]:
                        self.passed_all_points = True

    def estimate_speed(self, distances, ratios):
        speeds = []
        i = 0
        while i < len(ratios):
            current = i + 1
            accumulated = 0
            while current < len(self.anchors) - 1 \
                    and np.array_equal(self.anchors[current][1], self.anchors[current + 1][1]):
                accumulated += distances[current - 1]
                current += 1
            if accumulated != 0:
                current -= 1
            (a_time, a_pos), (b_time, b_pos) = self.anchors[current - 1], self.anchors[current]
            if np.array_equal(a_pos, b_pos):
                break
            pixels = np.sqrt((a_pos[0] - b_pos[0])**2 + (a_pos[1] - b_pos[1])**2)
            distance = accumulated + (pixels * ratios[current - 1])
            time = abs(a_time - b_time)
            speeds.append(distance/time)
            i = current + 1
        if not speeds:
            self.anchors.clear()
            self.direction = 0
            self.positions.clear()
            self.passed_all_points = False
            return -1
        self.evaluated = True
        speed = np.mean(speeds) * 3600/5280
        return speed








