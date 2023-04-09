from collections import OrderedDict
import numpy as np

'''
Algorithmic process derived from Adrian Rosebrock, PhD
'''


def calc_centroids(boxes):
    centroids = np.zeros((len(boxes), 2), dtype='int')
    trackers = [None for _ in range(len(boxes))]
    for (i, ((x, y, fx, fy), tracker)) in enumerate(boxes):
        a, b = (x + fx)//2, (y + fy)//2
        centroids[i] = [a, b]
        trackers[i] = tracker
    return centroids, trackers


class VehicleCache:

    def __init__(self):
        self.vehicles = OrderedDict()
        self.undetected = OrderedDict()
        self.nextId = 0
        self.clear_from_main = []
        self.trackers = {}

    def add(self, vehicle, tracker):
        self.vehicles[self.nextId] = vehicle
        self.undetected[self.nextId] = 0
        self.trackers[tracker] = self.nextId
        self.nextId += 1

    def remove(self, vehicle_id: int):
        del self.vehicles[vehicle_id]
        del self.undetected[vehicle_id]

    def get_undetected(self):
        undetected = self.clear_from_main.copy()
        self.clear_from_main.clear()
        return undetected

    def update(self, boxes, centroid_track):
        if not boxes:
            for vehicle in self.undetected.keys():
                self.undetected[vehicle] += 1
            return self.vehicles
        centroids, trackers = calc_centroids(boxes)
        if len(self.vehicles) == 0:
            for i in range(len(centroids)):
                self.add(centroids[i], trackers[i])
        elif centroid_track:
            self.determine_nearest_neighbors(centroids, trackers)
        else:
            self.update_positions(centroids, trackers)
        return self.vehicles

    def update_positions(self, centroids, trackers):
        for i in range(len(centroids)):
            if trackers[i] in self.trackers:
                vehicle_id = self.trackers[trackers[i]]
                self.vehicles[vehicle_id] = centroids[i]

    def determine_nearest_neighbors(self, centroids, trackers):
        prev_centroids = list(self.vehicles.values())
        x = np.array(prev_centroids)
        distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        ids = list(self.vehicles.keys())
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        used_rows, used_cols = self.determine_used_dimensions(ids, centroids, trackers,
                                                              rows, cols)
        return self.update_unused_dimensions(distances, used_rows, used_cols, ids, centroids, trackers)

    def determine_used_dimensions(self, ids, centroids, trackers, rows, cols):
        used_rows, used_cols = set(), set()
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            vehicle_id = ids[row]
            self.vehicles[vehicle_id] = centroids[col]
            self.trackers[trackers[col]] = vehicle_id
            self.undetected[vehicle_id] = 0
            used_rows.add(row)
            used_cols.add(col)
        return used_rows, used_cols

    def update_unused_dimensions(self, distances, used_rows, used_cols, ids, centroids, trackers):
        unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
        unused_cols = set(range(0, distances.shape[1])).difference(used_cols)
        if distances.shape[0] >= distances.shape[1]:
            for row in unused_rows:
                vehicle_id = ids[row]
                self.undetected[vehicle_id] += 1
                if self.undetected[vehicle_id] > 20:
                    self.clear_from_main.append(vehicle_id)
                    self.remove(vehicle_id)
        else:
            for col in unused_cols:
                self.add(centroids[col], trackers[col])






