from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np


def calc_centroids(boxes):
    centroids = np.zeros((len(boxes), 2))
    for (i, (x, y, fx, fy)) in enumerate(boxes):
        a, b = (x + fx)//2, (y + fy)//2
        centroids[i] = [a, b]
    return centroids


class VehicleCache:

    def __init__(self):
        self.vehicles = OrderedDict()
        self.undetected = OrderedDict()
        self.nextId = 0

    def add(self, vehicle):
        self.vehicles[self.nextId] = vehicle
        self.undetected[self.nextId] = 0
        self.nextId += 1

    def remove(self, vehicle_id):
        del self.vehicles[vehicle_id]
        del self.undetected[vehicle_id]

    def update(self, boxes):
        if not boxes:
            for vehicle in self.undetected.keys():
                self.undetected[vehicle] += 1
            return self.vehicles
        centroids = calc_centroids(boxes)
        if len(self.vehicles) == 0:
            for centroid in centroids:
                self.add(centroid)
        else:
            self.determine_nearest_neighbors(centroids)
        return self.vehicles

    def determine_nearest_neighbors(self, centroids):
        distances = dist.cdist(np.array(self.vehicles.values()), centroids)
        ids = list(self.vehicles.keys())
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            vehicle_id = ids[row]
            self.vehicles[vehicle_id] = centroids[col]
            self.undetected[vehicle_id] = 0
            used_rows.add(row)
            used_cols.add(col)
        unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
        unused_cols = set(range(0, distances.shape[1])).difference(used_cols)
        if distances.shape[0] >= distances.shape[1]:
            for row in unused_rows:
                vehicle_id = ids[row]
                self.undetected[vehicle_id] += 1
                if self.undetected[vehicle_id] > 50:
                    self.remove(vehicle_id)
        else:
            for col in unused_cols:
                self.add(centroids[col])






