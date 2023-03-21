from datetime import datetime


class Vehicle:

    def __init__(self, position, points):
        self.anchors = {}
        self.positions = [position]
        for i in range(points):
            self.anchors[i] = None
        self.direction = None

    def update_position(self, position, points):
        p = points.copy()
        if self.direction == -1:
            p = p[::-1]
        for (i, point) in enumerate(p):
            if position[0] > points[0] and not self.anchors[i]:
                self.anchors[i] = (datetime.now().timestamp(), position)








