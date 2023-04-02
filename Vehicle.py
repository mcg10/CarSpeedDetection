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

    def estimate_speed(self, ratios):
        speeds = []
        #print(self.anchors)
        for (i, ratio) in enumerate(ratios):
            if not self.anchors[i] or not self.anchors[i + 1]:
                continue
            (a_time, a_pos), (b_time, b_pos) = self.anchors[i], self.anchors[i + 1]
            if np.array_equal(a_pos, b_pos):
                continue
            pixels = np.sqrt((a_pos[0] - b_pos[0])**2 + (a_pos[1] - b_pos[1])**2)
            distance = pixels * ratio
            time = abs(a_time - b_time)
            speeds.append(distance/time)
        if not speeds:
            self.anchors.clear()
            self.direction = 0
            self.positions.clear()
            self.passed_all_points = False
            return -1
        self.evaluated = True
        speed = np.mean(speeds) * 3600/5280
        if speed > 0:
            print(self.anchors)
        return speed








