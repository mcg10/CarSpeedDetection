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
        if self.direction == -1:
            p = p[::-1]
            backwards = True
        for (i, point) in enumerate(p):
            if not self.anchors.get(i, None):
                if (not backwards and position[0] > point[0]) or (backwards and position[0] < point[0]):
                    self.anchors[i] = (datetime.now().timestamp(), position)
                    if point == p[-1]:
                        self.passed_all_points = True

    def estimate_speed(self, distances):
        return 0








